#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# PROBE Experiment Launcher
#
# Submits all experiment phases to SLURM.
#
# Usage:
#   bash slurm/launch_experiments.sh              # Full experiment suite
#   bash slurm/launch_experiments.sh --phase 2    # Phase 2 only (main training)
#   bash slurm/launch_experiments.sh --dry-run    # Show commands without submitting
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:---all}"
DRY_RUN=false
[[ "$PHASE" == "--dry-run" ]] && DRY_RUN=true && PHASE="--all"
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p slurm/logs

submit() {
    local desc="$1"; shift
    echo "  Submitting: $desc"
    if $DRY_RUN; then
        echo "    [DRY RUN] $*"
    else
        local out
        out=$("$@" 2>&1)
        echo "    $out"
    fi
}

echo "========================================"
echo "PROBE Experiment Launcher"
echo "========================================"
echo ""

# ── Phase 2: Main Training (3 methods x 1 seed, parallel on 4 GPUs) ──
if [[ "$PHASE" == "--all" || "$PHASE" == "--phase" && "${2:-}" == "2" || "$PHASE" == "2" ]]; then
    echo "Phase 2: Main Training (GRPO + DPO + GFlowNet, seed=42)"
    echo "  4 GPUs, ~48-72 hours"
    submit "GRPO+DPO+GFlowNet (seed=42)" \
        sbatch --export=STEPS=2000,SEED=42 slurm/train_all.sbatch
    echo ""
fi

# ── Phase 4: Ablations (5 runs, 2 GPUs each) ─────────────────────────
if [[ "$PHASE" == "--all" || "$PHASE" == "--phase" && "${2:-}" == "4" || "$PHASE" == "4" ]]; then
    echo "Phase 4: Ablations (GRPO, 1000 steps each)"
    echo "  2 GPUs each, ~24 hours each"

    submit "A1: No EMA" \
        sbatch --job-name=probe-A1-no-ema \
        --export=METHOD=grpo,STEPS=1000,SEED=42,RUN_NAME=ablation_no_ema \
        slurm/train_single.sbatch

    submit "A2: No curriculum" \
        sbatch --job-name=probe-A2-no-curriculum \
        --export=METHOD=grpo,STEPS=1000,SEED=42,RUN_NAME=ablation_no_curriculum \
        slurm/train_single.sbatch

    submit "A3: No novelty" \
        sbatch --job-name=probe-A3-no-novelty \
        --export=METHOD=grpo,STEPS=1000,SEED=42,RUN_NAME=ablation_no_novelty \
        slurm/train_single.sbatch

    submit "A4: No plausibility" \
        sbatch --job-name=probe-A4-no-plaus \
        --export=METHOD=grpo,STEPS=1000,SEED=42,RUN_NAME=ablation_no_plaus \
        slurm/train_single.sbatch

    submit "A5: No progress" \
        sbatch --job-name=probe-A5-no-progress \
        --export=METHOD=grpo,STEPS=1000,SEED=42,RUN_NAME=ablation_no_progress \
        slurm/train_single.sbatch
    echo ""
fi

# ── Phase 5: Seed Robustness (best method x 3 seeds) ─────────────────
if [[ "$PHASE" == "--all" || "$PHASE" == "--phase" && "${2:-}" == "5" || "$PHASE" == "5" ]]; then
    echo "Phase 5: Seed Robustness (GRPO x 3 seeds, 2000 steps each)"
    echo "  2 GPUs each, ~48 hours each"

    submit "GRPO seed=123" \
        sbatch --job-name=probe-grpo-s123 \
        --export=METHOD=grpo,STEPS=2000,SEED=123 \
        slurm/train_single.sbatch

    submit "GRPO seed=456" \
        sbatch --job-name=probe-grpo-s456 \
        --export=METHOD=grpo,STEPS=2000,SEED=456 \
        slurm/train_single.sbatch
    echo ""
fi

echo "========================================"
echo "Check status: squeue -u \$USER"
echo "Monitor logs: tail -f slurm/logs/*.log"
echo "========================================"
