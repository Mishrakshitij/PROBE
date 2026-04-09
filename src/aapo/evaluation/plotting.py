"""Publication-quality plotting for AAPO training and evaluation.

Uses scienceplots for Nature/IEEE style. All figures are saved as
high-resolution PDFs suitable for paper submission.

Usage:
    from aapo.evaluation.plotting import TrainingPlotter, EvalPlotter

    # Training curves from JSONL logs
    plotter = TrainingPlotter.from_jsonl("data/logs/grpo_seed42/training_metrics.jsonl")
    plotter.plot_all(save_dir="figures/")

    # Evaluation plots from report JSON
    eval_plotter = EvalPlotter.from_json("data/eval_report.json")
    eval_plotter.plot_all(save_dir="figures/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to load scienceplots; fall back gracefully
try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False


# ── Helpers ──────────────────────────────────────────────────────────

def _apply_style():
    """Apply publication style. Uses scienceplots if available."""
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "nature"])
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        })


def _smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average for smoother curves."""
    if len(values) < 2:
        return values
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300):
    """Save figure as both PDF (vector) and PNG (raster)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path.with_suffix(".pdf")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _extract(records: list[dict], key: str, default: float = 0.0) -> np.ndarray:
    """Extract a key from a list of dicts as a numpy array."""
    return np.array([r.get(key, default) for r in records], dtype=float)


# ── Color palettes ───────────────────────────────────────────────────

# Colorblind-friendly palette
COLORS = {
    "grpo": "#0072B2",
    "dpo": "#D55E00",
    "gflownet": "#009E73",
    "total": "#0072B2",
    "violation": "#D55E00",
    "progress": "#009E73",
    "novelty": "#CC79A7",
    "plausibility": "#E69F00",
    "kl": "#56B4E9",
    "grad": "#F0E442",
}

METHOD_LABELS = {
    "grpo": "GRPO",
    "dpo": "DPO",
    "gflownet": "GFlowNet",
}


# ══════════════════════════════════════════════════════════════════════
#  TrainingPlotter — plots from training_metrics.jsonl
# ══════════════════════════════════════════════════════════════════════

class TrainingPlotter:
    """Generate publication-quality training plots from JSONL logs."""

    def __init__(self, records: list[dict[str, Any]], method: str = ""):
        self.records = records
        self.steps = _extract(records, "step")
        self.method = method or (records[0].get("method", "") if records else "")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> TrainingPlotter:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records)

    # ── Individual plot methods ──────────────────────────────────────

    def plot_loss(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Training loss over steps."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        loss = _extract(self.records, "loss")
        ax.plot(self.steps, loss, alpha=0.2, color=COLORS["total"], linewidth=0.5)
        ax.plot(self.steps, _smooth(loss, smooth_window), color=COLORS["total"],
                linewidth=1.5, label="Total loss")

        if self.method == "grpo":
            pg = _extract(self.records, "pg_loss")
            kl = _extract(self.records, "kl_loss")
            ax.plot(self.steps, _smooth(pg, smooth_window), color=COLORS["violation"],
                    linewidth=1.2, linestyle="--", label="PG loss")
            ax.plot(self.steps, _smooth(kl, smooth_window), color=COLORS["kl"],
                    linewidth=1.2, linestyle=":", label="KL loss")

        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Training Loss ({METHOD_LABELS.get(self.method, self.method)})")
        _save_fig(fig, Path(save_dir) / "loss")

    def plot_rewards(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Reward components (mean with min-max shading) over training."""
        _apply_style()
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
        axes = axes.flatten()

        components = [
            ("reward_total", "Total Reward", COLORS["total"]),
            ("reward_violation", "Violation", COLORS["violation"]),
            ("reward_progress", "Progress", COLORS["progress"]),
            ("reward_novelty", "Novelty", COLORS["novelty"]),
            ("reward_plausibility", "Plausibility", COLORS["plausibility"]),
        ]

        for idx, (prefix, title, color) in enumerate(components):
            ax = axes[idx]
            mean = _extract(self.records, f"{prefix}_mean")
            mn = _extract(self.records, f"{prefix}_min")
            mx = _extract(self.records, f"{prefix}_max")

            ax.fill_between(self.steps, mn, mx, alpha=0.15, color=color)
            ax.plot(self.steps, mean, alpha=0.2, color=color, linewidth=0.5)
            ax.plot(self.steps, _smooth(mean, smooth_window), color=color, linewidth=1.5)
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("Reward")

        # Remove last empty subplot
        axes[-1].set_visible(False)

        for ax in axes[3:5]:
            ax.set_xlabel("Training step")

        fig.suptitle(f"Reward Components ({METHOD_LABELS.get(self.method, self.method)})",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        _save_fig(fig, Path(save_dir) / "rewards")

    def plot_grad_norm(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Gradient norm over training."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        gn = _extract(self.records, "grad_norm")
        ax.plot(self.steps, gn, alpha=0.2, color=COLORS["grad"], linewidth=0.5)
        ax.plot(self.steps, _smooth(gn, smooth_window), color="#333333", linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Gradient norm")
        ax.set_title("Gradient Norm")
        _save_fig(fig, Path(save_dir) / "grad_norm")

    def plot_kl_divergence(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """KL divergence over training."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        kl = _extract(self.records, "kl_divergence")
        ax.plot(self.steps, kl, alpha=0.2, color=COLORS["kl"], linewidth=0.5)
        ax.plot(self.steps, _smooth(kl, smooth_window), color=COLORS["kl"], linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("KL divergence")
        ax.set_title("KL Divergence from Reference")
        _save_fig(fig, Path(save_dir) / "kl_divergence")

    def plot_reward_schedule(self, save_dir: str | Path = "figures"):
        """Reward weight schedule (alpha, beta, gamma) over training."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        alpha = _extract(self.records, "alpha")
        beta = _extract(self.records, "beta")
        gamma = _extract(self.records, "gamma")

        ax.plot(self.steps, alpha, label=r"$\alpha$ (progress)", color=COLORS["progress"], linewidth=1.5)
        ax.plot(self.steps, beta, label=r"$\beta$ (novelty)", color=COLORS["novelty"], linewidth=1.5)
        ax.plot(self.steps, gamma, label=r"$\gamma$ (plausibility)", color=COLORS["plausibility"],
                linewidth=1.5, linestyle="--")

        ax.set_xlabel("Training step")
        ax.set_ylabel("Weight")
        ax.set_title("Reward Weight Schedule")
        ax.legend(frameon=False, fontsize=8)
        _save_fig(fig, Path(save_dir) / "reward_schedule")

    def plot_violation_yield(self, save_dir: str | Path = "figures", smooth_window: int = 30):
        """Violation yield over training."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        vy = _extract(self.records, "violation_yield")
        ax.plot(self.steps, vy, alpha=0.15, color=COLORS["violation"], linewidth=0.5)
        ax.plot(self.steps, _smooth(vy, smooth_window), color=COLORS["violation"], linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Violations / episode")
        ax.set_title("Violation Yield")
        _save_fig(fig, Path(save_dir) / "violation_yield")

    def plot_completion_length(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Completion length over training."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        mn = _extract(self.records, "completion_len_min")
        mean = _extract(self.records, "completion_len_mean")
        mx = _extract(self.records, "completion_len_max")

        ax.fill_between(self.steps, mn, mx, alpha=0.15, color=COLORS["total"])
        ax.plot(self.steps, _smooth(mean, smooth_window), color=COLORS["total"], linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Completion length (words)")
        ax.set_title("Completion Length")
        _save_fig(fig, Path(save_dir) / "completion_length")

    def plot_curriculum(self, save_dir: str | Path = "figures"):
        """Curriculum stage and k_max over training."""
        _apply_style()
        fig, ax1 = plt.subplots(figsize=(4, 3))

        stage = _extract(self.records, "curriculum_stage")
        kmax = _extract(self.records, "k_max")

        ax1.plot(self.steps, stage, color=COLORS["total"], linewidth=1.5, label="Stage")
        ax1.set_xlabel("Training step")
        ax1.set_ylabel("Stage", color=COLORS["total"])
        ax1.tick_params(axis="y", labelcolor=COLORS["total"])

        ax2 = ax1.twinx()
        ax2.plot(self.steps, kmax, color=COLORS["violation"], linewidth=1.5,
                 linestyle="--", label=r"$K_{\max}$")
        ax2.set_ylabel(r"$K_{\max}$ (turns)", color=COLORS["violation"])
        ax2.tick_params(axis="y", labelcolor=COLORS["violation"])

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)

        ax1.set_title("Curriculum Progress")
        _save_fig(fig, Path(save_dir) / "curriculum")

    def plot_replay_buffer(self, save_dir: str | Path = "figures"):
        """Replay buffer size over training."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        buf = _extract(self.records, "replay_buffer_size")
        ax.plot(self.steps, buf, color=COLORS["progress"], linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Buffer size")
        ax.set_title("Replay Buffer Size")
        _save_fig(fig, Path(save_dir) / "replay_buffer")

    def plot_advantages(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Advantage statistics (GRPO only)."""
        if self.method != "grpo":
            return
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        mn = _extract(self.records, "advantage_min")
        mean = _extract(self.records, "advantage_mean")
        mx = _extract(self.records, "advantage_max")

        ax.fill_between(self.steps, mn, mx, alpha=0.15, color=COLORS["total"])
        ax.plot(self.steps, mean, alpha=0.2, color=COLORS["total"], linewidth=0.5)
        ax.plot(self.steps, _smooth(mean, smooth_window), color=COLORS["total"], linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Advantage")
        ax.set_title("GRPO Advantages")
        _save_fig(fig, Path(save_dir) / "advantages")

    def plot_clip_fraction(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Clip fraction (GRPO only)."""
        if self.method != "grpo":
            return
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        cf = _extract(self.records, "clip_fraction")
        ax.plot(self.steps, cf, alpha=0.2, color=COLORS["violation"], linewidth=0.5)
        ax.plot(self.steps, _smooth(cf, smooth_window), color=COLORS["violation"], linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Clip fraction")
        ax.set_title("GRPO Clip Fraction")
        ax.set_ylim(-0.05, 1.05)
        _save_fig(fig, Path(save_dir) / "clip_fraction")

    def plot_log_probs(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Log-probability statistics over training."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        mn = _extract(self.records, "log_prob_min")
        mean = _extract(self.records, "log_prob_mean")
        mx = _extract(self.records, "log_prob_max")

        ax.fill_between(self.steps, mn, mx, alpha=0.15, color=COLORS["total"])
        ax.plot(self.steps, _smooth(mean, smooth_window), color=COLORS["total"], linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Log probability")
        ax.set_title("Log Probabilities")
        _save_fig(fig, Path(save_dir) / "log_probs")

    def plot_wallclock(self, save_dir: str | Path = "figures", smooth_window: int = 10):
        """Wall-clock time per step."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4, 3))

        wc = _extract(self.records, "wallclock_s")
        ax.plot(self.steps, wc, alpha=0.2, color="#666666", linewidth=0.5)
        ax.plot(self.steps, _smooth(wc, smooth_window), color="#333333", linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Wall-Clock Time per Step")
        _save_fig(fig, Path(save_dir) / "wallclock")

    def plot_all(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Generate all training plots."""
        sd = Path(save_dir)
        self.plot_loss(sd, smooth_window)
        self.plot_rewards(sd, smooth_window)
        self.plot_grad_norm(sd, smooth_window)
        self.plot_kl_divergence(sd, smooth_window)
        self.plot_reward_schedule(sd)
        self.plot_violation_yield(sd, smooth_window + 10)
        self.plot_completion_length(sd, smooth_window)
        self.plot_curriculum(sd)
        self.plot_replay_buffer(sd)
        self.plot_advantages(sd, smooth_window)
        self.plot_clip_fraction(sd, smooth_window)
        self.plot_log_probs(sd, smooth_window)
        self.plot_wallclock(sd)
        print(f"All training plots saved to {sd}/")


# ══════════════════════════════════════════════════════════════════════
#  Multi-run comparison plots
# ══════════════════════════════════════════════════════════════════════

class ComparisonPlotter:
    """Compare multiple training runs (e.g., GRPO vs DPO vs GFlowNet)."""

    def __init__(self):
        self.runs: dict[str, list[dict[str, Any]]] = {}

    def add_run(self, name: str, path: str | Path):
        """Add a run by loading its JSONL training log."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        self.runs[name] = records

    def plot_metric_comparison(
        self,
        metric: str,
        ylabel: str = "",
        title: str = "",
        save_dir: str | Path = "figures",
        filename: str = "comparison",
        smooth_window: int = 20,
    ):
        """Plot a single metric across all runs."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4.5, 3))

        for name, records in self.runs.items():
            steps = _extract(records, "step")
            vals = _extract(records, metric)
            method = records[0].get("method", name) if records else name
            color = COLORS.get(method, None)
            label = METHOD_LABELS.get(method, name)

            ax.plot(steps, vals, alpha=0.1, color=color, linewidth=0.5)
            ax.plot(steps, _smooth(vals, smooth_window), color=color,
                    linewidth=1.5, label=label)

        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel or metric)
        ax.set_title(title or metric.replace("_", " ").title())
        ax.legend(frameon=False, fontsize=8)
        _save_fig(fig, Path(save_dir) / filename)

    def plot_reward_comparison(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Compare total reward across runs with shaded min-max bands."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(4.5, 3))

        for name, records in self.runs.items():
            steps = _extract(records, "step")
            mean = _extract(records, "reward_total_mean")
            mn = _extract(records, "reward_total_min")
            mx = _extract(records, "reward_total_max")
            method = records[0].get("method", name) if records else name
            color = COLORS.get(method, None)
            label = METHOD_LABELS.get(method, name)

            ax.fill_between(steps, mn, mx, alpha=0.08, color=color)
            ax.plot(steps, _smooth(mean, smooth_window), color=color,
                    linewidth=1.5, label=label)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Total reward")
        ax.set_title("Total Reward Comparison")
        ax.legend(frameon=False, fontsize=8)
        _save_fig(fig, Path(save_dir) / "reward_comparison")

    def plot_all_comparisons(self, save_dir: str | Path = "figures", smooth_window: int = 20):
        """Generate standard comparison plots."""
        sd = Path(save_dir)

        self.plot_reward_comparison(sd, smooth_window)

        self.plot_metric_comparison(
            "loss", ylabel="Loss", title="Training Loss Comparison",
            save_dir=sd, filename="loss_comparison", smooth_window=smooth_window,
        )
        self.plot_metric_comparison(
            "violation_yield", ylabel="Violations / episode",
            title="Violation Yield Comparison",
            save_dir=sd, filename="yield_comparison", smooth_window=smooth_window + 10,
        )
        self.plot_metric_comparison(
            "grad_norm", ylabel="Gradient norm",
            title="Gradient Norm Comparison",
            save_dir=sd, filename="grad_norm_comparison", smooth_window=smooth_window,
        )
        self.plot_metric_comparison(
            "n_unique_policies", ylabel="Unique policies triggered",
            title="Policy Discovery Comparison",
            save_dir=sd, filename="policy_discovery_comparison", smooth_window=smooth_window,
        )
        print(f"All comparison plots saved to {sd}/")


# ══════════════════════════════════════════════════════════════════════
#  EvalPlotter — plots from evaluation report JSON
# ══════════════════════════════════════════════════════════════════════

class EvalPlotter:
    """Generate publication-quality evaluation plots from report JSON."""

    def __init__(self, report: dict[str, Any]):
        self.report = report

    @classmethod
    def from_json(cls, path: str | Path) -> EvalPlotter:
        with open(path) as f:
            return cls(json.load(f))

    def plot_coverage_curve(self, save_dir: str | Path = "figures"):
        """Coverage curve: cumulative policy coverage over episodes."""
        _apply_style()
        curve = self.report.get("coverage_curve", [])
        if not curve:
            return

        fig, ax = plt.subplots(figsize=(4, 3))
        episodes = np.arange(1, len(curve) + 1)
        coverage_pct = np.array(curve) * 100

        ax.plot(episodes, coverage_pct, color=COLORS["total"], linewidth=1.5)
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Policy coverage (%)")
        ax.set_title("Policy Coverage Curve")
        ax.set_ylim(0, 105)
        ax.axhline(100, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        _save_fig(fig, Path(save_dir) / "coverage_curve")

    def plot_per_policy_counts(self, save_dir: str | Path = "figures"):
        """Bar chart of violations per policy."""
        _apply_style()
        coverage = self.report.get("coverage", {})
        counts = coverage.get("per_policy_counts", {})
        if not counts:
            return

        fig, ax = plt.subplots(figsize=(6, 3))
        policies = sorted(counts.keys())
        values = [counts[p] for p in policies]

        # Color by category
        category_colors = {
            "B": COLORS["total"],
            "S": COLORS["violation"],
            "P": COLORS["novelty"],
            "O": COLORS["progress"],
        }
        colors = [category_colors.get(p[0], "#666666") for p in policies]

        ax.bar(range(len(policies)), values, color=colors, alpha=0.8, width=0.7)
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policies, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Violation count")
        ax.set_title("Violations per Policy")
        fig.tight_layout()
        _save_fig(fig, Path(save_dir) / "per_policy_counts")

    def plot_severity_distribution(self, save_dir: str | Path = "figures"):
        """Pie or bar chart of violation severity distribution."""
        _apply_style()
        yield_data = self.report.get("yield", {})
        per_sev = yield_data.get("per_severity", {})
        if not per_sev:
            return

        fig, ax = plt.subplots(figsize=(3.5, 3))
        severity_order = ["critical", "high", "medium", "low"]
        severity_colors = ["#D55E00", "#E69F00", "#56B4E9", "#999999"]

        labels = []
        sizes = []
        colors = []
        for sev, col in zip(severity_order, severity_colors):
            if sev in per_sev:
                labels.append(sev.capitalize())
                sizes.append(per_sev[sev])
                colors.append(col)

        ax.bar(range(len(labels)), sizes, color=colors, alpha=0.85, width=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count")
        ax.set_title("Violations by Severity")
        _save_fig(fig, Path(save_dir) / "severity_distribution")

    def plot_all(self, save_dir: str | Path = "figures"):
        """Generate all evaluation plots."""
        sd = Path(save_dir)
        self.plot_coverage_curve(sd)
        self.plot_per_policy_counts(sd)
        self.plot_severity_distribution(sd)
        print(f"All evaluation plots saved to {sd}/")
