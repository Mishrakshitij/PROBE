#!/bin/bash
#SBATCH --job-name=aapo-eval-tb
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodelist=mbz-titan-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/eval_%x_%j.out
#SBATCH --error=slurm/logs/eval_%x_%j.err

set -euo pipefail

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/lora/dir}"
EPISODES="${EPISODES:-50}"
MAX_TURNS="${MAX_TURNS:-14}"
SEED="${SEED:-42}"
EVAL_TAG="${EVAL_TAG:-}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_mbzuai.yaml}"

AAPO_DIR="/shared/home/kshitij.mishra/aapo"
CONDA_BASE="/shared/home/kshitij.mishra/miniconda3"
ENV_BIN="${CONDA_BASE}/envs/aapo/bin"
HF_HOME="/shared/home/kshitij.mishra/.cache/huggingface"
TARGET_MODEL="/shared/home/nils.lukas/models/Qwen3.6-35B-A3B"

cd "$AAPO_DIR"
mkdir -p slurm/logs data/eval

export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== AAPO Eval — tau-bench | Checkpoint=$CHECKPOINT Episodes=$EPISODES MaxTurns=$MAX_TURNS | Job=$SLURM_JOB_ID Node=$SLURM_NODELIST | $(date) ==="

# Unique port per job to avoid collisions when multiple jobs share a node
AAPO_SERVER_PORT=$((8001 + SLURM_JOB_ID % 1000))
export AAPO_SERVER_PORT

# GPU 0 = vLLM target+judge  |  GPU 1 = auditor
VLLM_LOG="slurm/logs/vllm_eval_${SLURM_JOB_ID}.log"
echo "[1/2] Starting SGLang server ($TARGET_MODEL) on GPU 0 port $AAPO_SERVER_PORT..."
VLLM_PID=$(bash slurm/helpers/start_vllm_mbzuai.sh \
    "$TARGET_MODEL" 0 "$AAPO_SERVER_PORT" "$VLLM_LOG")
echo "  SGLang PID=$VLLM_PID"

"${ENV_BIN}/python" slurm/helpers/wait_vllm.py "$AAPO_SERVER_PORT" 600 \
    || { echo "ERROR: SGLang failed — see $VLLM_LOG"; kill "$VLLM_PID" 2>/dev/null; exit 1; }

echo "[2/2] Running evaluation..."
"${ENV_BIN}/python" scripts/eval_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --env tau_bench \
    --tau-bench-path vendor/tau-bench \
    --episodes "$EPISODES" \
    --max-turns "$MAX_TURNS" \
    --models-config "$MODELS_CONFIG" \
    --auditor-gpu 0 \
    --seed "$SEED"
RC=$?

kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true
echo "=== Eval exit=$RC | $(date) ==="
exit $RC
