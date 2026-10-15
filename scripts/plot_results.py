#!/usr/bin/env python3
"""Generate publication-quality plots from training logs and evaluation reports.

Usage:
    # Plot a single training run
    python scripts/plot_results.py --training-log data/logs/grpo_seed42/training_metrics.jsonl

    # Compare multiple runs
    python scripts/plot_results.py --compare \
        grpo=data/logs/grpo_seed42/training_metrics.jsonl \
        dpo=data/logs/dpo_seed42/training_metrics.jsonl \
        gflownet=data/logs/gflownet_seed42/training_metrics.jsonl

    # Plot evaluation results
    python scripts/plot_results.py --eval-report data/eval_report.json

    # All together
    python scripts/plot_results.py \
        --training-log data/logs/grpo_seed42/training_metrics.jsonl \
        --eval-report data/eval_report.json \
        --save-dir figures/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from probe.evaluation.plotting import TrainingPlotter, ComparisonPlotter, EvalPlotter


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from PROBE logs"
    )
    parser.add_argument(
        "--training-log", type=str, nargs="*", default=[],
        help="Path(s) to training_metrics.jsonl file(s). Each gets its own set of plots."
    )
    parser.add_argument(
        "--compare", type=str, nargs="*", default=[],
        help="Comparison runs as name=path pairs (e.g., grpo=data/logs/grpo/training_metrics.jsonl)"
    )
    parser.add_argument(
        "--eval-report", type=str, default=None,
        help="Path to evaluation report JSON"
    )
    parser.add_argument(
        "--save-dir", type=str, default="figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--smooth", type=int, default=20,
        help="Smoothing window for EMA curves"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution for raster outputs"
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Single-run training plots
    for log_path in args.training_log:
        print(f"\nGenerating training plots from: {log_path}")
        plotter = TrainingPlotter.from_jsonl(log_path)
        method = plotter.method or Path(log_path).parent.name
        run_dir = save_dir / method
        plotter.plot_all(save_dir=run_dir, smooth_window=args.smooth)

    # Multi-run comparison
    if args.compare:
        print("\nGenerating comparison plots...")
        comp = ComparisonPlotter()
        for entry in args.compare:
            if "=" in entry:
                name, path = entry.split("=", 1)
            else:
                name = Path(entry).parent.name
                path = entry
            print(f"  Adding run: {name} <- {path}")
            comp.add_run(name, path)
        comp.plot_all_comparisons(save_dir=save_dir / "comparison", smooth_window=args.smooth)

    # Evaluation plots
    if args.eval_report:
        print(f"\nGenerating evaluation plots from: {args.eval_report}")
        eval_plotter = EvalPlotter.from_json(args.eval_report)
        eval_plotter.plot_all(save_dir=save_dir / "evaluation")

    if not args.training_log and not args.compare and not args.eval_report:
        parser.print_help()
        return

    print(f"\nAll figures saved to: {save_dir}/")


if __name__ == "__main__":
    main()
