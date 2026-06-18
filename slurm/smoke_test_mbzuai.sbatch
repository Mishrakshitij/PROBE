#!/bin/bash
#SBATCH --job-name=aapo-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/smoke_%j.out
#SBATCH --error=slurm/logs/smoke_%j.err

set -euo pipefail

AAPO_DIR="/shared/home/kshitij.mishra/aapo"
CONDA_BASE="/shared/home/kshitij.mishra/miniconda3"
ENV_BIN="${CONDA_BASE}/envs/aapo/bin"
HF_HOME="/shared/home/kshitij.mishra/.cache/huggingface"
TARGET_MODEL="/shared/home/nils.lukas/models/Qwen3.6-35B-A3B"

cd "$AAPO_DIR"
mkdir -p slurm/logs data/logs

export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== AAPO Smoke Test — tau-bench | Job=$SLURM_JOB_ID | $(date) ==="

# vLLM server on GPU 0 (only GPU needed — smoke test uses ScriptedAuditor)
VLLM_LOG="slurm/logs/vllm_smoke_${SLURM_JOB_ID}.log"
echo "[1/2] Starting vLLM server ($TARGET_MODEL)..."
VLLM_PID=$(bash slurm/helpers/start_vllm_mbzuai.sh \
    "$TARGET_MODEL" 0 8001 "$VLLM_LOG")
echo "  vLLM PID=$VLLM_PID"

"${ENV_BIN}/python" slurm/helpers/wait_vllm.py 8001 600 \
    || { echo "ERROR: vLLM failed to start — see $VLLM_LOG"; kill "$VLLM_PID" 2>/dev/null; exit 1; }

echo "[2/2] Running tau-bench smoke test (scripted auditor, 5 turns)..."
"${ENV_BIN}/python" scripts/smoke_test.py \
    --env tau_bench \
    --tau-bench-path vendor/tau-bench \
    --models-config configs/models_mbzuai.yaml
RC=$?

kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true
echo "=== Smoke test exit=$RC | $(date) ==="
exit $RC
