#!/usr/bin/env python3
"""Run the rebuttal comparison baselines on SkyAssist.

Three baselines requested by the NeurIPS reviewers / AC (Priority 1 & the
"simpler non-learning baselines" concern):

  scripted  — static compliance suite: 20 handcrafted per-policy conversations,
              replayed turn-by-turn without adapting to the target. This is the
              "τ-bench-style fixed scenario" / non-learning baseline.
  coop      — cooperative LLM user-simulator (non-adversarial static baseline).
  fewshot   — few-shot-prompted general-purpose prober (no RL training).

Scoring is identical to scripts/eval_checkpoint.py (same verifier suite, same
LLM judge for B4/B6/S5/P1, same CompositeReward, same clustering and
ReportGenerator) so the numbers drop straight into Table 1 alongside the
trained auditors and the B0/B2 baselines. Coverage = fraction of 20 policies
triggered; yield = violations / 100 episodes.

Usage (on a GPU node with the vLLM 30B target on port 8001):
    python scripts/run_rebuttal_baselines.py --baseline scripted \
        --episodes 50 --max-turns 8 --seed 42 \
        --models-config configs/models_slurm.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapo.environment.database import FlightDB
from aapo.environment.tools import ToolRegistry
from aapo.environment.policies import PolicyRegistry
from aapo.environment.target_agent import TargetAgent
from aapo.environment.logger import InteractionLogger
from aapo.reward.attribution import TurnRewardAttributor
from aapo.reward.judge import LLMJudge
from aapo.reward.novelty import NoveltyScorer
from aapo.reward.composite import CompositeReward
from aapo.replay.embedder import TrajectoryEmbedder
from aapo.replay.cluster import ClusterManager
from aapo.evaluation.report import ReportGenerator
from aapo.trajectory.episode_runner import EpisodeRunner
from aapo.auditor.scenarios import ScenarioPool
from aapo.auditor.baselines import (
    ScriptedAuditor,
    CooperativeAuditor,
    FewShotAuditor,
    SCRIPTED_CONVERSATIONS,
    build_generic_conversations,
)
from aapo.inference.model_loader import load_inference_backend, load_models_config


# scripted = static suite WITH oracle (1 hand-crafted attack per policy)
# fixed    = static suite WITHOUT oracle (generic booking flows, random routes)
# coop     = cooperative LLM user-simulator
# fewshot  = few-shot-prompted general-purpose prober
BASELINES = ("scripted", "fixed", "coop", "fewshot")
STATIC_BASELINES = ("scripted", "fixed")


def main():
    parser = argparse.ArgumentParser(description="Run rebuttal baselines on SkyAssist")
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--db-path", type=str, default="data/flights.db")
    parser.add_argument("--models-config", type=str, default="configs/models_slurm.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_name = f"baseline_{args.baseline}"
    if args.output is None:
        args.output = f"data/eval/{run_name}_report.json"
    if args.log_dir is None:
        args.log_dir = f"data/eval/{run_name}_logs"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Environment
    db = FlightDB(args.db_path)
    tools = ToolRegistry(db)
    policies = PolicyRegistry()

    # Inference backends (target on the shared vLLM server; judge shares it)
    target_client = load_inference_backend("target", args.models_config)
    judge_client = load_inference_backend("judge", args.models_config)
    models_cfg = load_models_config(args.models_config)
    target_mcfg = models_cfg.get("models", {}).get("target", {})

    target = TargetAgent(
        tools, client=target_client,
        max_context_tokens=target_mcfg.get("max_model_len", 0),
        reserve_tokens=target_mcfg.get("max_tokens", 512),
    )
    judge = LLMJudge(client=judge_client)
    logger = InteractionLogger(args.log_dir)

    # Scoring — identical configuration to eval_checkpoint.py
    attributor = TurnRewardAttributor(db, policies, judge=judge)
    novelty = NoveltyScorer(policies.policy_ids)
    reward = CompositeReward(
        db=db, policy_registry=policies,
        attributor=attributor, novelty_scorer=novelty,
    )
    runner = EpisodeRunner(target=target, logger=None, max_turns=args.max_turns)

    # Build the requested baseline auditor
    scenario_pool = ScenarioPool(seed=args.seed)
    scripted_auditor = None
    if args.baseline == "scripted":
        scripted_auditor = ScriptedAuditor(SCRIPTED_CONVERSATIONS)
        auditor = scripted_auditor
    elif args.baseline == "fixed":
        convs = build_generic_conversations(args.db_path, seed=args.seed, n=20)
        if not convs:
            raise RuntimeError("No valid routes found to build generic conversations")
        scripted_auditor = ScriptedAuditor(convs)
        auditor = scripted_auditor
    elif args.baseline == "coop":
        # Cooperative simulator uses the target's backend (local vLLM 30B)
        auditor = CooperativeAuditor(client=load_inference_backend("target", args.models_config))
    elif args.baseline == "fewshot":
        auditor = FewShotAuditor(client=load_inference_backend("target", args.models_config))
    else:  # pragma: no cover
        raise ValueError(args.baseline)

    print(f"Running baseline '{args.baseline}': {args.episodes} episodes, "
          f"max_turns={args.max_turns}, seed={args.seed}")
    print(f"  Target: {target_client.model_name}")
    print(f"  Output: {args.output}")
    if scripted_auditor is not None:
        kind = "policy-targeted (oracle)" if args.baseline == "scripted" else "generic no-oracle"
        print(f"  Scripts: {len(scripted_auditor.conversations)} {kind} conversations "
              f"(cycled to fill {args.episodes} episodes)")
    print()

    episodes = []
    for i in range(args.episodes):
        if scripted_auditor is not None:
            # Static suites (scripted/fixed): cycle deterministically through the
            # canned conversations. The stochastic target yields varied traces.
            scripted_auditor.reset(i)
            ep = runner.run(auditor, episode_id=f"{run_name}_{i:04d}")
        else:
            # Cooperative / few-shot simulators get a concrete scenario per
            # episode from the SAME held-out pool the trained auditors use at
            # eval — only the auditor policy differs.
            scenario = scenario_pool.sample()
            auditor.set_scenario(scenario.description)
            ep = runner.run(auditor, episode_id=f"{run_name}_{i:04d}")

        reward.score_episode(ep)
        logger.log_episode(ep)
        episodes.append(ep)

        n_viol = len(ep.violations)
        policies_hit = sorted(set(v.policy_id for v in ep.violations))
        if (i + 1) % 10 == 0 or i == 0:
            total_viols = sum(len(e.violations) for e in episodes)
            all_policies = set(v.policy_id for e in episodes for v in e.violations)
            tag = f"[{scripted_auditor.current.target_policy}] " if scripted_auditor is not None else ""
            print(f"  Episode {i+1}/{args.episodes} | {tag}this_ep: {n_viol} {policies_hit} | "
                  f"cumul: {total_viols} violations, {len(all_policies)}/20 policies")

    # Report — same generator as the trained-auditor eval
    print("\nGenerating evaluation report...")
    embedder = TrajectoryEmbedder()
    cluster_mgr = ClusterManager(embedder)
    cluster_mgr.add_and_cluster(episodes)

    reporter = ReportGenerator(policies.policy_ids)
    report = reporter.generate(episodes, cluster_mgr)
    report["baseline"] = args.baseline
    reporter.print_summary(report)
    reporter.export(report, args.output)
    print(f"\nReport saved to {args.output}")
    print(f"Episode logs saved to {args.log_dir}")

    db.close()


if __name__ == "__main__":
    main()
