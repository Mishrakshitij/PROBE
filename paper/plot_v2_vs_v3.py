#!/usr/bin/env python3
"""V2 vs V3 training-time comparison from JSONL logs.

Generates fig_v2_vs_v3_training.pdf showing how V2 and V3 reward designs
affect training dynamics across GRPO, DPO, GFlowNet.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/home/salem.lahlou/kmishra/aapo")
LOG = BASE / "data" / "logs"
OUT = BASE / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
})

COL_V2 = "#1f77b4"  # blue
COL_V3 = "#d62728"  # red


def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    return [seen[k] for k in sorted(seen)]


def extract(rows, field):
    steps, vals = [], []
    for r in rows:
        if field in r and r[field] is not None:
            try:
                v = float(r[field])
                steps.append(r["step"])
                vals.append(v)
            except (TypeError, ValueError):
                pass
    return np.array(steps), np.array(vals)


def rolling(arr, win=10):
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - win + 1)
        out[i] = np.mean(arr[lo:i + 1])
    return out


METHODS = [("grpo", "GRPO"), ("dpo", "DPO"), ("gflownet", "GFlowNet")]
VERSIONS = [("v2", "Behavior reward", COL_V2), ("v3", "Calibrated reward", COL_V3)]


def fig_yield_and_length():
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5), sharey="row")

    for col, (method, mlabel) in enumerate(METHODS):
        # Top row: violation yield
        ax_y = axes[0, col]
        ax_y.set_title(f"{mlabel}")
        for ver, vlabel, c in VERSIONS:
            path = LOG / f"{method}_seed42_{ver}" / "training_metrics.jsonl"
            if not path.exists():
                continue
            rows = load_jsonl(path)
            s, y = extract(rows, "violation_yield")
            if len(s) == 0:
                continue
            ax_y.plot(s, rolling(y, 15), label=vlabel, color=c, alpha=0.95)
            ax_y.fill_between(s, 0, rolling(y, 15), color=c, alpha=0.08)
        if col == 0:
            ax_y.set_ylabel("Violation yield (per step, rolling-15)")
        ax_y.legend(loc="upper left", frameon=True)
        ax_y.set_xlim(left=0)
        ax_y.set_ylim(bottom=0)

        # Bottom row: completion length
        ax_l = axes[1, col]
        for ver, vlabel, c in VERSIONS:
            path = LOG / f"{method}_seed42_{ver}" / "training_metrics.jsonl"
            if not path.exists():
                continue
            rows = load_jsonl(path)
            s, l = extract(rows, "completion_len_mean")
            if len(s) == 0:
                continue
            ax_l.plot(s, rolling(l, 15), label=vlabel, color=c, alpha=0.95)
        if col == 0:
            ax_l.set_ylabel("Completion length (words, rolling-15)")
        ax_l.set_xlabel("Training step")
        ax_l.legend(loc="upper right", frameon=True)
        ax_l.set_xlim(left=0)

    fig.suptitle("Behavior vs Calibrated reward: training dynamics across methods", y=1.02, fontsize=13)
    fig.tight_layout()
    out = OUT / "fig_v2_vs_v3_training.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")
    return out


def summary_table():
    """Print a numerical summary at fixed checkpoints (steps 50, 150, 250)."""
    print("\n" + "=" * 75)
    print(f"{'Method':<14} {'Ver':<3} {'Step':>5} {'Yield (avg)':>13} {'Len (avg)':>11} {'Loss':>8}")
    print("-" * 75)
    for method, mlabel in METHODS:
        for ver in ["v2", "v3"]:
            path = LOG / f"{method}_seed42_{ver}" / "training_metrics.jsonl"
            if not path.exists():
                continue
            rows = load_jsonl(path)
            for step in [50, 150, 250]:
                # Average over a 20-step window centered at the target
                window = [r for r in rows if step - 10 <= r["step"] <= step + 10]
                if not window:
                    continue
                ys = [r.get("violation_yield", 0) for r in window if r.get("violation_yield") is not None]
                ls = [r.get("completion_len_mean", 0) for r in window if r.get("completion_len_mean") is not None]
                losses = [r.get("loss", 0) for r in window if r.get("loss") is not None]
                if not ys:
                    continue
                print(f"{mlabel:<14} {ver.upper():<3} {step:>5d} {np.mean(ys):>13.2f} "
                      f"{np.mean(ls) if ls else 0:>11.1f} {np.mean(losses) if losses else 0:>8.3f}")
        print("-" * 75)


if __name__ == "__main__":
    out = fig_yield_and_length()
    summary_table()
    print(f"\nGenerated: {out}")
