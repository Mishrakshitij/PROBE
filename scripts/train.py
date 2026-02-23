#!/usr/bin/env python3
"""Main training entry point.

Expects vLLM server(s) already running (see scripts/launch_servers.py).
Auditor loaded locally with optional QLoRA on configured GPU.

Usage:
    python scripts/train.py [--method grpo] [--steps 1000] [--seed 42]
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
from aapo.replay.buffer import ReplayBuffer
from aapo.training.curriculum import CurriculumScheduler
from aapo.training.loop import TrainingLoop
from aapo.inference.model_loader import load_inference_backend, load_auditor_config


def main():
    parser = argparse.ArgumentParser(description="Train AAPO auditor")
    parser.add_argument("--method", choices=["grpo", "dpo", "gflownet"], default="grpo")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", type=str, default="data/flights.db")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config()

    # Environment
    db = FlightDB(args.db_path)
    tools = ToolRegistry(db)
    policies = PolicyRegistry()

    # Inference backends from models.yaml
    target_client = load_inference_backend("target", args.models_config)
    judge_client = load_inference_backend("judge", args.models_config)

    target = TargetAgent(tools, client=target_client)
    judge = LLMJudge(client=judge_client)
    logger = InteractionLogger()

    # Auditor with QLoRA
    auditor_cfg = load_auditor_config(args.models_config)
    lora_cfg = auditor_cfg.get("lora")
    auditor = AuditorPolicy(
        model_name=auditor_cfg.get("name", "Qwen/Qwen3-4B-Instruct"),
        device=f"cuda:{auditor_cfg['gpu']}" if "gpu" in auditor_cfg else None,
        max_new_tokens=auditor_cfg.get("max_new_tokens", 256),
        temperature=auditor_cfg.get("temperature", 0.8),
        quantize_4bit=auditor_cfg.get("quantization") == "bitsandbytes-4bit",
        lora_config=lora_cfg,
    )

    # Reward
    attributor = TurnRewardAttributor(db, policies, judge=judge)
    novelty = NoveltyScorer(policies.policy_ids)
    reward = CompositeReward(
        db=db, policy_registry=policies,
        attributor=attributor, novelty_scorer=novelty,
    )

    # Replay + Curriculum
    replay = ReplayBuffer(max_size=cfg.replay.max_size)
    curriculum = CurriculumScheduler(
        k_max_stages=cfg.training.curriculum.k_max_stages,
    )

    # Training
    loop = TrainingLoop(
        auditor=auditor, target=target, db=db,
        policy_registry=policies, reward=reward,
        replay=replay, curriculum=curriculum,
        logger=logger, method=args.method,
        group_size=args.group_size, batch_size=args.batch_size,
        seed=args.seed,
    )

    ckpt_dir = Path(args.checkpoint_dir)

    def print_progress(step, metrics):
        if step % 10 == 0:
            loss = metrics.get("loss", "n/a")
            vyield = metrics.get("violation_yield", 0)
            stage = metrics.get("stage", 0)
            print(f"Step {step:4d} | loss={loss:.4f} | yield={vyield:.3f} | stage={stage}")
        if step > 0 and step % args.checkpoint_every == 0 and auditor._has_lora:
            ckpt_path = ckpt_dir / f"step_{step}"
            auditor.save_lora(ckpt_path)
            print(f"  -> Saved LoRA checkpoint to {ckpt_path}")

    print(f"Starting training: method={args.method}, steps={args.steps}, seed={args.seed}")
    print(f"  Target: {target_client.model_name}")
    print(f"  Auditor: {auditor_cfg.get('name', 'default')}")
    metrics = loop.train(num_steps=args.steps, callback=print_progress)
    print(f"\nTraining complete. {len(metrics)} steps recorded.")

    # Save final checkpoint
    if auditor._has_lora:
        final_path = ckpt_dir / "final"
        auditor.save_lora(final_path)
        print(f"Final LoRA saved to {final_path}")

    db.close()


if __name__ == "__main__":
    main()
