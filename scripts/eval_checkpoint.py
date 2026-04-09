#!/usr/bin/env python3
"""Evaluate an auditor LoRA checkpoint against the target agent.

Loads the checkpoint, runs N episodes via vLLM target, scores violations,
and generates a full evaluation report.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint data/checkpoints/dpo_seed42/step_25 \
        --episodes 100 --max-turns 8 \
        --models-config configs/models_slurm.yaml --auditor-gpu 1 \
        --output data/eval/dpo_step25.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapo.config import load_config
from aapo.environment.database import FlightDB
from aapo.environment.tools import ToolRegistry
from aapo.environment.policies import PolicyRegistry
from aapo.environment.target_agent import TargetAgent
from aapo.environment.logger import InteractionLogger
from aapo.auditor.model import AuditorPolicy
from aapo.reward.attribution import TurnRewardAttributor
from aapo.reward.judge import LLMJudge
from aapo.reward.novelty import NoveltyScorer
from aapo.reward.composite import CompositeReward
from aapo.replay.embedder import TrajectoryEmbedder
from aapo.replay.cluster import ClusterManager
from aapo.evaluation.report import ReportGenerator
from aapo.trajectory.episode_runner import EpisodeRunner
from aapo.inference.model_loader import load_inference_backend, load_auditor_config, load_models_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate AAPO auditor checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Max turns per episode")
    parser.add_argument("--db-path", type=str, default="data/flights.db")
    parser.add_argument("--models-config", type=str, default="configs/models_slurm.yaml")
    parser.add_argument("--auditor-gpu", type=int, default=1)
    parser.add_argument("--output", type=str, default=None,
                        help="Output report JSON path (default: auto from checkpoint)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory to save episode logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        ckpt_name = ckpt_path.parent.name + "_" + ckpt_path.name
        args.output = f"data/eval/{ckpt_name}_report.json"
    if args.log_dir is None:
        ckpt_name = ckpt_path.parent.name + "_" + ckpt_path.name
        args.log_dir = f"data/eval/{ckpt_name}_logs"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)

    # Environment
    db = FlightDB(args.db_path)
    tools = ToolRegistry(db)
    policies = PolicyRegistry()

    # Inference backends
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

    # Auditor: load base model with QLoRA + LoRA (same as training), then load weights
    auditor_cfg = load_auditor_config(args.models_config)
    lora_cfg = auditor_cfg.get("lora")
    auditor = AuditorPolicy(
        model_name=auditor_cfg.get("name", "Qwen/Qwen3-4B-Instruct"),
        device=f"cuda:{args.auditor_gpu}",
        max_new_tokens=auditor_cfg.get("max_new_tokens", 256),
        temperature=auditor_cfg.get("temperature", 0.8),
        quantize_4bit=auditor_cfg.get("quantization") == "bitsandbytes-4bit",
        lora_config=lora_cfg,
    )
    # Load trained adapter weights into the existing PeftModel
    print(f"Loading LoRA checkpoint from {ckpt_path}")
    from safetensors.torch import load_file as load_safetensors
    state_dict = load_safetensors(str(ckpt_path / "adapter_model.safetensors"))
    auditor.model.load_state_dict(state_dict, strict=False)
    auditor.model.eval()

    # Reward scoring
    attributor = TurnRewardAttributor(db, policies, judge=judge)
    novelty = NoveltyScorer(policies.policy_ids)
    reward = CompositeReward(
        db=db, policy_registry=policies,
        attributor=attributor, novelty_scorer=novelty,
    )

    # Episode runner (no logger — we log after scoring to include violations)
    runner = EpisodeRunner(target=target, logger=None, max_turns=args.max_turns)
    from aapo.auditor.scenarios import ScenarioPool
    scenario_pool = ScenarioPool(seed=args.seed)

    print(f"Running {args.episodes} evaluation episodes (max_turns={args.max_turns})...")
    print(f"  Target: {target_client.model_name}")
    print(f"  Auditor: {auditor_cfg.get('name')} + LoRA from {ckpt_path}")
    print(f"  Output: {args.output}")
    print()

    episodes = []
    for i in range(args.episodes):
        scenario = scenario_pool.sample()
        auditor.set_scenario(scenario.description)
        ep = runner.run(auditor)
        # Score violations (populates ep.violations)
        reward.score_episode(ep)
        # Re-log with violations included (runner logs before scoring)
        logger.log_episode(ep)
        episodes.append(ep)

        n_viol = len(ep.violations)
        policies_hit = set(v.policy_id for v in ep.violations)
        if (i + 1) % 10 == 0 or i == 0:
            total_viols = sum(len(e.violations) for e in episodes)
            all_policies = set(v.policy_id for e in episodes for v in e.violations)
            print(f"  Episode {i+1}/{args.episodes} | "
                  f"this_ep: {n_viol} violations {policies_hit} | "
                  f"cumul: {total_viols} violations, {len(all_policies)}/20 policies")

    # Generate report
    print("\nGenerating evaluation report...")
    embedder = TrajectoryEmbedder()
    cluster_mgr = ClusterManager(embedder)
    cluster_mgr.add_and_cluster(episodes)

    reporter = ReportGenerator(policies.policy_ids)
    report = reporter.generate(episodes, cluster_mgr)
    reporter.print_summary(report)
    reporter.export(report, args.output)
    print(f"\nReport saved to {args.output}")
    print(f"Episode logs saved to {args.log_dir}")

    db.close()


if __name__ == "__main__":
    main()
