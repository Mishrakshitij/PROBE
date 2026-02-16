#!/usr/bin/env python3
"""Main training entry point.

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
from aapo.reward.novelty import NoveltyScorer
from aapo.reward.composite import CompositeReward
from aapo.replay.buffer import ReplayBuffer
from aapo.training.curriculum import CurriculumScheduler
from aapo.training.loop import TrainingLoop


def main():
    parser = argparse.ArgumentParser(description="Train AAPO auditor")
    parser.add_argument("--method", choices=["grpo", "dpo", "gflownet"], default="grpo")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", type=str, default="data/flights.db")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config()

    # Environment
    db = FlightDB(args.db_path)
    tools = ToolRegistry(db)
    policies = PolicyRegistry()
    target = TargetAgent(tools, model=cfg.target.model)
    logger = InteractionLogger()

    # Auditor
    model_name = args.model or cfg.training.auditor_model
    auditor = AuditorPolicy(model_name=model_name)

    # Reward
    attributor = TurnRewardAttributor(db, policies)
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

    def print_progress(step, metrics):
        if step % 10 == 0:
            loss = metrics.get("loss", "n/a")
            vyield = metrics.get("violation_yield", 0)
            stage = metrics.get("stage", 0)
            print(f"Step {step:4d} | loss={loss:.4f} | yield={vyield:.3f} | stage={stage}")

    print(f"Starting training: method={args.method}, steps={args.steps}, seed={args.seed}")
    metrics = loop.train(num_steps=args.steps, callback=print_progress)
    print(f"\nTraining complete. {len(metrics)} steps recorded.")

    db.close()


if __name__ == "__main__":
    main()
