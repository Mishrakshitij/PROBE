#!/usr/bin/env python3
"""Generate all publication-grade figures for the AAPO NeurIPS 2026 paper.

Outputs PDF files to paper/figures/.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/salem.lahlou/kmishra/aapo")
LOG_DIR = BASE / "data" / "logs"
EVAL_DIR = BASE / "data" / "eval"
FIG_DIR = BASE / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style configuration (no scienceplots dependency)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 9,
    "legend.framealpha": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "text.usetex": False,
})

# Consistent color palette
COLORS = {
    "GRPO": "#1f77b4",
    "DPO": "#d62728",
    "GFlowNet": "#2ca02c",
    "V1": "#8c564b",
    "V2": "#ff7f0e",
    "V3": "#9467bd",
}

METHOD_LABELS = {
    "grpo": "GRPO",
    "dpo": "DPO",
    "gflownet": "GFlowNet",
}

ALL_POLICIES = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "S1", "S2", "S3", "S4", "S5", "S6",
    "P1", "P2", "P3", "P4",
    "O1", "O2", "O3", "O4",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning list of dicts. Skips empty/bad lines."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip bad lines
    return records


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def rolling_mean(arr, window=5):
    """Simple rolling mean with minimum 1 valid point."""
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = np.nanmean(arr[lo : i + 1])
    return out


def deduplicate_steps(records: list[dict]) -> list[dict]:
    """If multiple records share the same step, keep only the last one."""
    seen = {}
    for r in records:
        seen[r["step"]] = r
    return [seen[k] for k in sorted(seen.keys())]


def extract_field(records, field):
    """Extract (steps, values) from records, skipping missing."""
    steps, vals = [], []
    for r in records:
        if field in r and r[field] is not None:
            steps.append(r["step"])
            vals.append(r[field])
    return np.array(steps), np.array(vals, dtype=float)


def training_log_path(method, version):
    """Return path to training_metrics.jsonl for a given method/version."""
    if version == "v1" or version == "":
        subdir = f"{method}_seed42"
    else:
        subdir = f"{method}_seed42_{version}"
    p = LOG_DIR / subdir / "training_metrics.jsonl"
    if not p.exists() and version == "v1":
        # Fallback: try without version suffix
        p = LOG_DIR / f"{method}_seed42" / "training_metrics.jsonl"
    return p


def eval_report_path(method, version, step):
    return EVAL_DIR / f"{method}_seed42_{version}_step_{step}_report.json"


def eval_episodes_path(method, version, step):
    return EVAL_DIR / f"{method}_seed42_{version}_step_{step}_logs" / "episodes.jsonl"


generated = []


# ===================================================================
# Figure 2: Coverage curves
# ===================================================================
def fig_coverage_curves():
    print("Generating fig_coverage_curves.pdf ...")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    entries = [
        ("GRPO", "grpo_seed42_v2_step_50_report.json"),
        ("DPO", "dpo_seed42_v2_step_50_report.json"),
        ("GFlowNet", "gflownet_seed42_v2_step_50_report.json"),
    ]

    for label, fname in entries:
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        report = load_json(path)
        curve = report.get("coverage_curve", [])
        if not curve:
            continue
        episodes = np.arange(1, len(curve) + 1)
        ax.plot(episodes, curve, label=f"{label} ({curve[-1]*100:.0f}%)",
                color=COLORS[label], linewidth=2)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative Policy Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, 50)
    ax.legend(loc="lower right", frameon=True)
    ax.set_title("Coverage Curves (V2, 50 episodes)")

    out = FIG_DIR / "fig_coverage_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    generated.append(out.name)
    print(f"  -> {out}")


# ===================================================================
# Figure 3: Training dynamics (3 panels)
# ===================================================================
def fig_training_dynamics():
    print("Generating fig_training_dynamics.pdf ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    # Panel (a): DPO V3 loss
    ax = axes[0]
    path = training_log_path("dpo", "v3")
    if path.exists():
        records = deduplicate_steps(load_jsonl(path))
        steps, loss = extract_field(records, "loss")
        ax.plot(steps, rolling_mean(loss, 5), color=COLORS["DPO"], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("DPO Loss")
        ax.set_title("(a) DPO V3 Loss")
    else:
        ax.set_title("(a) DPO V3 Loss [no data]")

    # Panel (b): Violation yield comparison (V3, all methods)
    ax = axes[1]
    for method, label in METHOD_LABELS.items():
        path = training_log_path(method, "v3")
        if not path.exists():
            continue
        records = deduplicate_steps(load_jsonl(path))
        steps, vy = extract_field(records, "violation_yield")
        if len(steps) > 0:
            ax.plot(steps, rolling_mean(vy, 5), label=label,
                    color=COLORS[label], linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Violation Yield")
    ax.set_title("(b) Violation Yield (V3)")
    ax.legend(loc="best", frameon=True)

    # Panel (c): Completion length across DPO versions
    ax = axes[2]
    for version, vlabel in [("v1", "V1"), ("v2", "V2"), ("v3", "V3")]:
        path = training_log_path("dpo", version)
        if not path.exists():
            continue
        records = deduplicate_steps(load_jsonl(path))
        steps, clen = extract_field(records, "completion_len_mean")
        if len(steps) > 0:
            ax.plot(steps, rolling_mean(clen, 5), label=f"DPO {vlabel}",
                    color=COLORS[vlabel], linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Completion Length (tokens)")
    ax.set_title("(c) Completion Length Evolution")
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    out = FIG_DIR / "fig_training_dynamics.pdf"
    fig.savefig(out)
    plt.close(fig)
    generated.append(out.name)
    print(f"  -> {out}")


# ===================================================================
# Figure 4: Reward engineering evolution bar chart
# ===================================================================
def fig_reward_evolution():
    print("Generating fig_reward_evolution.pdf ...")

    # Data: [avg_completion_len, coverage(/20), role_confusion(%), b4s5b6_share(%)]
    # V3 values are placeholders (marked with hatching)
    labels = ["V1", "V2", "V3"]
    metrics = ["Avg Completion\nLength (tokens)", "Coverage\n(/20 policies)",
               "Role Confusion\nRate (%)", "B4/S5/B6\nShare (%)"]
    data = np.array([
        [130, 9, 58, 85],   # V1
        [60, 12, 11, 83],   # V2
        [45, 11, 5, 76],    # V3 (placeholder)
    ], dtype=float)

    v3_placeholder = True  # mark V3 values

    n_groups = len(labels)
    n_metrics = len(metrics)
    x = np.arange(n_groups)
    width = 0.18
    offsets = np.arange(n_metrics) - (n_metrics - 1) / 2
    bar_colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]

    fig, ax = plt.subplots(figsize=(7, 4))

    for j in range(n_metrics):
        bars = ax.bar(x + offsets[j] * width, data[:, j], width * 0.9,
                       label=metrics[j], color=bar_colors[j], edgecolor="white",
                       linewidth=0.5)
        # Add hatching for V3 (placeholder)
        if v3_placeholder:
            bars[2].set_hatch("//")
            bars[2].set_edgecolor("gray")

        # Annotate values
        for i, bar in enumerate(bars):
            val = data[i, j]
            suffix = "*" if (i == 2 and v3_placeholder) else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}{suffix}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("Reward Engineering Evolution Across Versions")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    # Footnote
    ax.annotate("* V3 values are preliminary estimates",
                xy=(0.01, -0.12), xycoords="axes fraction",
                fontsize=7, fontstyle="italic", color="gray")

    fig.tight_layout()
    out = FIG_DIR / "fig_reward_evolution.pdf"
    fig.savefig(out)
    plt.close(fig)
    generated.append(out.name)
    print(f"  -> {out}")


# ===================================================================
# Figure 5: Per-policy violation heatmap
# ===================================================================
def fig_policy_heatmap():
    print("Generating fig_policy_heatmap.pdf ...")

    methods_info = [
        ("GRPO", "grpo_seed42_v2_step_50_report.json"),
        ("DPO", "dpo_seed42_v2_step_50_report.json"),
        ("GFlowNet", "gflownet_seed42_v2_step_50_report.json"),
    ]

    method_names = []
    matrix = []

    for label, fname in methods_info:
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue

        report = load_json(path)
        ppc = report.get("coverage", {}).get("per_policy_counts", {})

        row = []
        for pol in ALL_POLICIES:
            row.append(ppc.get(pol, 0))
        matrix.append(row)
        method_names.append(label)

    if not matrix:
        print("  [SKIP] No data for heatmap")
        return

    matrix = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 2.8))

    # Log scale coloring
    log_matrix = np.log10(1 + matrix)
    im = ax.imshow(log_matrix, cmap="YlOrRd", aspect="auto",
                   interpolation="nearest")

    # Annotate cells
    for i in range(len(method_names)):
        for j in range(len(ALL_POLICIES)):
            count = int(matrix[i, j])
            text_color = "white" if log_matrix[i, j] > 1.5 else "black"
            ax.text(j, i, str(count), ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold" if count > 0 else "normal")

    ax.set_xticks(np.arange(len(ALL_POLICIES)))
    ax.set_xticklabels(ALL_POLICIES, fontsize=9)
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)
    ax.set_xlabel("Policy")
    ax.set_title("Per-Policy Violation Counts (V2, 50 episodes)")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("log$_{10}$(1 + count)", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "fig_policy_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    generated.append(out.name)
    print(f"  -> {out}")


# ===================================================================
# Figure A1: DPO V3 margin growth
# ===================================================================
def fig_dpo_margins():
    print("Generating fig_dpo_margins.pdf ...")
    path = training_log_path("dpo", "v3")
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    records = deduplicate_steps(load_jsonl(path))
    steps_m, margin = extract_field(records, "margin_mean")
    steps_a, acc = extract_field(records, "implicit_accuracy")

    fig, ax1 = plt.subplots(figsize=(6, 3.5))

    color_margin = COLORS["DPO"]
    color_acc = "#ff7f0e"

    if len(steps_m) > 0:
        ax1.plot(steps_m, rolling_mean(margin, 5), color=color_margin,
                 linewidth=1.5, label="Margin (mean)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Margin Mean", color=color_margin)
    ax1.tick_params(axis="y", labelcolor=color_margin)

    ax2 = ax1.twinx()
    if len(steps_a) > 0:
        ax2.plot(steps_a, rolling_mean(acc, 5), color=color_acc,
                 linewidth=1.5, linestyle="--", label="Implicit Accuracy")
    ax2.set_ylabel("Implicit Accuracy", color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True)

    ax1.set_title("DPO V3: Margin Growth and Implicit Accuracy")
    fig.tight_layout()
    out = FIG_DIR / "fig_dpo_margins.pdf"
    fig.savefig(out)
    plt.close(fig)
    generated.append(out.name)
    print(f"  -> {out}")


# ===================================================================
# Figure A2: GFlowNet log_Z and TB loss (2 panels)
# ===================================================================
def fig_gflownet_logz():
    print("Generating fig_gflownet_logz.pdf ...")
    path = training_log_path("gflownet", "v3")
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return

    records = deduplicate_steps(load_jsonl(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Panel (a): log_Z
    ax = axes[0]
    steps, logz = extract_field(records, "log_Z")
    if len(steps) > 0:
        ax.plot(steps, rolling_mean(logz, 5), color=COLORS["GFlowNet"], linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("log Z")
    ax.set_title("(a) GFlowNet V3: log Z")

    # Panel (b): TB loss (gflownet_loss or loss)
    ax = axes[1]
    steps_l, loss = extract_field(records, "loss")
    if len(steps_l) > 0:
        ax.plot(steps_l, rolling_mean(loss, 5), color=COLORS["GFlowNet"], linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("TB Loss")
    ax.set_title("(b) GFlowNet V3: Trajectory Balance Loss")

    fig.tight_layout()
    out = FIG_DIR / "fig_gflownet_logz.pdf"
    fig.savefig(out)
    plt.close(fig)
    generated.append(out.name)
    print(f"  -> {out}")


# ===================================================================
# Figure A3: Curriculum stage progression
# ===================================================================
def fig_curriculum():
    print("Generating fig_curriculum.pdf ...")
    fig, ax = plt.subplots(figsize=(6, 3.5))

    has_data = False
    for method, label in METHOD_LABELS.items():
        path = training_log_path(method, "v3")
        if not path.exists():
            continue
        records = deduplicate_steps(load_jsonl(path))
        steps, kmax = extract_field(records, "k_max")
        if len(steps) > 0:
            ax.step(steps, kmax, where="post", label=label,
                    color=COLORS[label], linewidth=1.8)
            has_data = True

    if not has_data:
        print("  [SKIP] No V3 data found for curriculum plot")
        plt.close(fig)
        return

    ax.set_xlabel("Training Step")
    ax.set_ylabel("$k_{\\mathrm{max}}$ (max conversation turns)")
    ax.set_title("Curriculum Progression (V3)")
    ax.legend(loc="best", frameon=True)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    out = FIG_DIR / "fig_curriculum.pdf"
    fig.savefig(out)
    plt.close(fig)
    generated.append(out.name)
    print(f"  -> {out}")


# ===================================================================
# Main
# ===================================================================
def main():
    print(f"Output directory: {FIG_DIR}\n")

    generators = [
        ("Figure 2", fig_coverage_curves),
        ("Figure 3", fig_training_dynamics),
        ("Figure 4", fig_reward_evolution),
        ("Figure 5", fig_policy_heatmap),
        ("Figure A1", fig_dpo_margins),
        ("Figure A2", fig_gflownet_logz),
        ("Figure A3", fig_curriculum),
    ]

    for name, func in generators:
        try:
            func()
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 60)
    print(f"Generated {len(generated)}/{len(generators)} figures:")
    for fname in generated:
        print(f"  {FIG_DIR / fname}")
    if len(generated) < len(generators):
        print(f"\n  ({len(generators) - len(generated)} figures skipped due to missing data)")


if __name__ == "__main__":
    main()
