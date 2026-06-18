#!/bin/bash
#SBATCH --job-name=aapo-train-tb
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodelist=mbz-titan-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/train_%x_%j.out
#SBATCH --error=slurm/logs/train_%x_%j.err

set -euo pipefail

METHOD="${METHOD:?Set METHOD=grpo|dpo|gflownet}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
CONTINUATION_TURNS="${CONTINUATION_TURNS:-3}"
RUN_TAG="${RUN_TAG:-}"
RESUME_FROM="${RESUME_FROM:-}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_mbzuai.yaml}"
RUN_NAME="${METHOD}_seed${SEED}${RUN_TAG}"

AAPO_DIR="/shared/home/kshitij.mishra/aapo"
CONDA_BASE="/shared/home/kshitij.mishra/miniconda3"
ENV_BIN="${CONDA_BASE}/envs/aapo/bin"
HF_HOME="/shared/home/kshitij.mishra/.cache/huggingface"
TARGET_MODEL="/shared/home/nils.lukas/models/Qwen3.6-35B-A3B"

cd "$AAPO_DIR"
mkdir -p slurm/logs data/logs "data/checkpoints/${RUN_NAME}"

export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== AAPO Training — ${METHOD^^} tau-bench | Job=$SLURM_JOB_ID Node=$SLURM_NODELIST Steps=$STEPS Seed=$SEED Run=$RUN_NAME | $(date) ==="

# Unique port per job to avoid collisions when multiple jobs share a node
AAPO_SERVER_PORT=$((8001 + SLURM_JOB_ID % 1000))
export AAPO_SERVER_PORT

# GPU 0 = vLLM target+judge (shared server)  |  GPU 1 = auditor (transformers+LoRA)
VLLM_LOG="slurm/logs/vllm_${RUN_NAME}_${SLURM_JOB_ID}.log"
echo "[1/2] Starting SGLang server ($TARGET_MODEL) on GPU 0 port $AAPO_SERVER_PORT..."
VLLM_PID=$(bash slurm/helpers/start_vllm_mbzuai.sh \
    "$TARGET_MODEL" 0 "$AAPO_SERVER_PORT" "$VLLM_LOG")
echo "  SGLang PID=$VLLM_PID"

"${ENV_BIN}/python" slurm/helpers/wait_vllm.py "$AAPO_SERVER_PORT" 600 \
    || { echo "ERROR: SGLang failed — see $VLLM_LOG"; kill "$VLLM_PID" 2>/dev/null; exit 1; }

echo "[2/2] Starting training (method=$METHOD, steps=$STEPS, seed=$SEED)..."
"${ENV_BIN}/python" scripts/train.py \
    --method "$METHOD" \
    --env tau_bench \
    --tau-bench-path vendor/tau-bench \
    --steps "$STEPS" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --group-size "$GROUP_SIZE" \
    --models-config "$MODELS_CONFIG" \
    --auditor-gpu 1 \
    --checkpoint-dir "data/checkpoints/${RUN_NAME}" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --log-dir "data/logs/${RUN_NAME}" \
    --run-name "$RUN_NAME" \
    --print-every 10 \
    --continuation-turns "$CONTINUATION_TURNS" \
    ${RESUME_FROM:+--resume-from "$RESUME_FROM"}
RC=$?

kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true
echo "=== Training exit=$RC | $(date) ==="
exit $RC
