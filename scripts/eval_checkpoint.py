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

    # tau-bench environment:
    python scripts/eval_checkpoint.py \
        --checkpoint data/checkpoints/dpo_seed42_taubench_v3/step_300 \
        --env tau_bench --tau-bench-path vendor/tau-bench \
        --episodes 50 --max-turns 14 \
        --models-config configs/models_mbzuai.yaml --auditor-gpu 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from probe.auditor.model import AuditorPolicy
from probe.replay.embedder import TrajectoryEmbedder
from probe.replay.cluster import ClusterManager
from probe.evaluation.report import ReportGenerator
from probe.inference.model_loader import load_inference_backend, load_auditor_config, load_models_config


def _load_checkpoint(auditor, ckpt_path: Path):
    """Load LoRA adapter weights using PEFT's key-aware loader.

    CRITICAL: must use peft's set_peft_model_state_dict() because the saved
    adapter file uses keys like "base_model.model....lora_A.weight" but the
    live PeftModel uses "base_model.model....lora_A.default.weight" (with
    adapter name). torch.load_state_dict(strict=False) silently drops these.
    """
    print(f"Loading LoRA checkpoint from {ckpt_path}")
    from safetensors.torch import load_file as load_safetensors
    from peft.utils.save_and_load import set_peft_model_state_dict
    state_dict = load_safetensors(str(ckpt_path / "adapter_model.safetensors"))
    load_result = set_peft_model_state_dict(auditor.model, state_dict)
    # Sanity check: assert no parameters were silently skipped
    if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
        print(f"  WARNING: {len(load_result.unexpected_keys)} unexpected keys")
    if hasattr(load_result, "missing_keys"):
        # peft's missing_keys excludes non-LoRA params; should be empty for adapter
        adapter_missing = [k for k in load_result.missing_keys if "lora_" in k]
        if adapter_missing:
            print(f"  ERROR: {len(adapter_missing)} adapter keys missing! "
                  f"First: {adapter_missing[0]}")
    print(f"  -> Loaded {len(state_dict)} adapter tensors")
    auditor.model.eval()


def _build_skyassist(args):
    from probe.environment.database import FlightDB
    from probe.environment.tools import ToolRegistry
    from probe.environment.policies import PolicyRegistry
    from probe.environment.target_agent import TargetAgent
    from probe.environment.logger import InteractionLogger
    from probe.reward.attribution import TurnRewardAttributor
    from probe.reward.judge import LLMJudge
    from probe.reward.novelty import NoveltyScorer
    from probe.reward.composite import CompositeReward
    from probe.trajectory.episode_runner import EpisodeRunner
    from probe.auditor.scenarios import ScenarioPool

    db = FlightDB(args.db_path)
    tools = ToolRegistry(db)
    policies = PolicyRegistry()

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
    attributor = TurnRewardAttributor(db, policies, judge=judge)
    novelty = NoveltyScorer(policies.policy_ids)
    reward = CompositeReward(
        db=db, policy_registry=policies,
        attributor=attributor, novelty_scorer=novelty,
    )
    runner = EpisodeRunner(target=target, logger=None, max_turns=args.max_turns)
    scenario_pool = ScenarioPool(seed=args.seed)

    return dict(
        db=db, target_client=target_client, policies=policies,
        reward=reward, runner=runner, scenario_pool=scenario_pool,
        logger=logger, n_policies=len(policies.policy_ids),
    )


def _build_tau_bench(args):
    from probe.environment.tau_bench_target import TauBenchTarget, TauBenchEpisodeRunner
    from probe.environment.logger import InteractionLogger
    from probe.reward.verifiers_taubench import (
        ALL_TB_VERIFIERS, TB_HYBRID_POLICY_IDS, TB_POLICY_SPECS,
        make_taubench_policy_registry,
    )
    from probe.reward.attribution import TurnRewardAttributor
    from probe.reward.novelty import NoveltyScorer
    from probe.reward.composite import CompositeReward
    from probe.auditor.scenarios import ScenarioPool

    target_client = load_inference_backend("target", args.models_config)
    policies = make_taubench_policy_registry()
    target = TauBenchTarget(
        client=target_client,
        task_split="test",
        tau_bench_path=args.tau_bench_path,
    )
    print(f"  tau-bench tasks loaded: {target.num_tasks()}")

    attributor = TurnRewardAttributor(
        db=None, policy_registry=policies, judge=None,
        verifiers=ALL_TB_VERIFIERS, hybrid_policy_ids=TB_HYBRID_POLICY_IDS,
    )
    novelty = NoveltyScorer(list(TB_POLICY_SPECS.keys()))
    reward = CompositeReward(
        db=None, policy_registry=policies,
        attributor=attributor, novelty_scorer=novelty,
    )
    scenario_pool = ScenarioPool.from_tau_bench(
        tau_bench_path=args.tau_bench_path, seed=args.seed
    )
    logger = InteractionLogger(args.log_dir)
    runner = TauBenchEpisodeRunner(
        target=target,
        scenario_pool=scenario_pool,
        logger=None,
        max_turns=args.max_turns,
    )

    return dict(
        db=None, target_client=target_client, policies=policies,
        reward=reward, runner=runner, scenario_pool=scenario_pool,
        logger=logger, n_policies=len(TB_POLICY_SPECS),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate PROBE auditor checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Evaluate the BASE auditor model zero-shot (no LoRA "
                             "weights loaded). Same pipeline/scenarios/verifiers as a "
                             "checkpoint eval — the fair same-model +/-training control.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Name for output/log dirs (required with --no-adapter)")
    parser.add_argument("--env", choices=["skyassist", "tau_bench"], default="skyassist")
    parser.add_argument("--tau-bench-path", type=str, default=None,
                        help="Path to tau-bench repo root (tau_bench env only)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Max turns per episode")
    parser.add_argument("--db-path", type=str, default="data/flights.db")
    parser.add_argument("--models-config", type=str, default="configs/models_slurm.yaml")
    parser.add_argument("--auditor-gpu", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_adapter:
        ckpt_path = None
        if not args.tag:
            print("ERROR: --tag is required with --no-adapter")
            sys.exit(1)
        ckpt_name = args.tag
    else:
        if not args.checkpoint:
            print("ERROR: --checkpoint is required (or pass --no-adapter)")
            sys.exit(1)
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        ckpt_name = ckpt_path.parent.name + "_" + ckpt_path.name

    if args.output is None:
        args.output = f"data/eval/{ckpt_name}_report.json"
    if args.log_dir is None:
        args.log_dir = f"data/eval/{ckpt_name}_logs"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)

    # Build environment
    env = _build_tau_bench(args) if args.env == "tau_bench" else _build_skyassist(args)
    target_client = env["target_client"]
    reward = env["reward"]
    runner = env["runner"]
    scenario_pool = env["scenario_pool"]
    logger = env["logger"]
    db = env["db"]
    n_policies = env["n_policies"]

    # Auditor
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
    if args.no_adapter:
        # Zero-shot control: peft zero-initialises lora_B, so an unloaded adapter is
        # an exact identity and the model is numerically the untrained base — running
        # through the same quantization, scenarios and scoring as a checkpoint eval.
        # Verify that identity rather than assuming it.
        import torch as _torch
        nonzero = [n for n, p in auditor.model.named_parameters()
                   if "lora_B" in n and _torch.count_nonzero(p).item() > 0]
        if nonzero:
            print(f"ERROR: {len(nonzero)} lora_B tensors are non-zero; the adapter is "
                  f"not an identity. First: {nonzero[0]}")
            sys.exit(1)
        n_lora = sum(1 for n, _ in auditor.model.named_parameters() if "lora_" in n)
        print(f"NO ADAPTER: base model zero-shot ({n_lora} LoRA tensors verified identity)")
        auditor.model.eval()
    else:
        _load_checkpoint(auditor, ckpt_path)

    print(f"Running {args.episodes} evaluation episodes (env={args.env}, max_turns={args.max_turns})...")
    print(f"  Target: {target_client.model_name}")
    if args.no_adapter:
        print(f"  Auditor: {auditor_cfg.get('name')} (zero-shot, no adapter)")
    else:
        print(f"  Auditor: {auditor_cfg.get('name')} + LoRA from {ckpt_path}")
    print(f"  Output: {args.output}")
    print()

    episodes = []
    for i in range(args.episodes):
        scenario = scenario_pool.sample()
        auditor.set_scenario(scenario.description)
        ep = runner.run(auditor)
        reward.score_episode(ep)
        logger.log_episode(ep)
        episodes.append(ep)

        n_viol = len(ep.violations)
        policies_hit = set(v.policy_id for v in ep.violations)
        if (i + 1) % 10 == 0 or i == 0:
            total_viols = sum(len(e.violations) for e in episodes)
            all_policies = set(v.policy_id for e in episodes for v in e.violations)
            print(f"  Episode {i+1}/{args.episodes} | "
                  f"this_ep: {n_viol} violations {policies_hit} | "
                  f"cumul: {total_viols} violations, {len(all_policies)}/{n_policies} policies")

    # Generate report
    print("\nGenerating evaluation report...")
    embedder = TrajectoryEmbedder()
    cluster_mgr = ClusterManager(embedder)
    cluster_mgr.add_and_cluster(episodes)

    policy_ids = list(env["policies"].policy_ids) if hasattr(env["policies"], "policy_ids") else []
    reporter = ReportGenerator(policy_ids)
    report = reporter.generate(episodes, cluster_mgr)
    reporter.print_summary(report)
    reporter.export(report, args.output)
    print(f"\nReport saved to {args.output}")
    print(f"Episode logs saved to {args.log_dir}")

    if db is not None:
        db.close()


if __name__ == "__main__":
    main()
