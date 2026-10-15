#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Quick monitor for running PROBE experiments
#
# Usage:
#   bash slurm/monitor.sh          # Latest step from each run
#   bash slurm/monitor.sh --tail   # Live tail of all training logs
#   bash slurm/monitor.sh --gpu    # GPU utilization on running nodes
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:---status}"

case "$MODE" in
    --status|-s)
        echo "========================================"
        echo "PROBE Training Status"
        echo "========================================"
        echo ""
        echo "SLURM Jobs:"
        squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.4D %R" 2>/dev/null || echo "  (no jobs)"
        echo ""

        # Show latest step from each training log
        for logdir in data/logs/*/; do
            if [ -f "${logdir}training_metrics.jsonl" ]; then
                name=$(basename "$logdir")
                last=$(tail -1 "${logdir}training_metrics.jsonl" 2>/dev/null)
                if [ -n "$last" ]; then
                    step=$(echo "$last" | python -c "import sys,json; d=json.load(sys.stdin); print(f'step={d[\"step\"]:5d} loss={d.get(\"loss\",0):.4f} R={d.get(\"reward_total_mean\",0):+.3f} yield={d.get(\"violation_yield\",0):.3f} grad={d.get(\"grad_norm\",0):.3f} stg={d.get(\"curriculum_stage\",0)}')" 2>/dev/null)
                    echo "  $name: $step"
                fi
            fi
        done
        ;;

    --tail|-t)
        echo "Tailing all training logs (Ctrl+C to stop)..."
        tail -f slurm/logs/grpo_*.log slurm/logs/dpo_*.log slurm/logs/gflownet_*.log 2>/dev/null || \
        tail -f data/logs/*/training_metrics.jsonl 2>/dev/null || \
        echo "No logs found yet."
        ;;

    --gpu|-g)
        echo "GPU utilization on job nodes:"
        for node in $(squeue -u "$USER" -h -o "%N" 2>/dev/null | sort -u); do
            echo ""
            echo "  Node: $node"
            srun --jobid="$(squeue -u "$USER" -h -o "%i" -w "$node" | head -1)" \
                nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        done
        ;;

    *)
        echo "Usage: bash slurm/monitor.sh [--status|--tail|--gpu]"
        ;;
esac
