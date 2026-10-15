#!/bin/bash
#SBATCH --job-name=aapo-train
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_%x_%j.out
#SBATCH --error=slurm/logs/train_%x_%j.err

set -euo pipefail

METHOD="${METHOD:?Set METHOD=grpo|dpo|gflownet}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_slurm.yaml}"
RUN_TAG="${RUN_TAG:-}"
RUN_NAME="${METHOD}_seed${SEED}${RUN_TAG}"
# Reward preset + per-weight overrides (reward-sensitivity study, Xtwt W3).
REWARD_PRESET="${REWARD_PRESET:-v3}"
GAMMA="${GAMMA:-}"
BETA_START="${BETA_START:-}"
BETA_END="${BETA_END:-}"

# Required env vars (set by user before sbatch):
#   AAPO_HOME, AAPO_CONDA_ENV, HF_HOME
cd "${AAPO_HOME:-$PWD}"
export PATH="${AAPO_CONDA_ENV:-/opt/conda}/bin:$PATH"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${AAPO_CONDA_ENV:-/opt/conda}"

mkdir -p slurm/logs data/logs data/checkpoints "$HF_HUB_CACHE"

# Install perf dependencies (idempotent, fast if already cached)
pip install --quiet liger-kernel flash-attn --no-build-isolation 2>/dev/null || true

echo "AAPO Training — ${METHOD^^} | Job=$SLURM_JOB_ID Node=$SLURM_NODELIST Steps=$STEPS Seed=$SEED Run=$RUN_NAME | $(date)"

[ ! -f data/flights.db ] && python scripts/init_db.py --seed "$SEED"

# GPU 0 = vLLM target, GPU 1 = auditor
VLLM_PID=$(bash slurm/helpers/start_vllm.sh "Qwen/Qwen3-30B-A3B-Instruct-2507" 0 8001 "slurm/logs/vllm_${METHOD}_${SLURM_JOB_ID}.log")
python slurm/helpers/wait_vllm.py 8001 600 || { kill $VLLM_PID 2>/dev/null; exit 1; }

TRAIN_ARGS=(--method "$METHOD" --steps "$STEPS" --seed "$SEED"
            --batch-size "$BATCH_SIZE" --group-size "$GROUP_SIZE"
            --models-config "$MODELS_CONFIG" --auditor-gpu 1
            --checkpoint-dir "data/checkpoints/${RUN_NAME}"
            --checkpoint-every "$CHECKPOINT_EVERY"
            --log-dir "data/logs/${RUN_NAME}"
            --run-name "$RUN_NAME" --print-every 10
            --continuation-turns 3
            --reward-preset "$REWARD_PRESET")
[ -n "$GAMMA" ]      && TRAIN_ARGS+=(--gamma "$GAMMA")
[ -n "$BETA_START" ] && TRAIN_ARGS+=(--beta-start "$BETA_START")
[ -n "$BETA_END" ]   && TRAIN_ARGS+=(--beta-end "$BETA_END")

python scripts/train.py "${TRAIN_ARGS[@]}"
RC=$?

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
echo "DONE: ${METHOD^^} exit=$RC | $(date)"
exit $RC
