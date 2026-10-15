#!/bin/bash
#SBATCH --job-name=probe-eval-transfer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/eval_transfer_%j.out
#SBATCH --error=slurm/logs/eval_transfer_%j.err

# Transfer experiment: trained DPO auditor vs GPT-4o-mini target.
# Requires:
#   CHECKPOINT   — path to LoRA checkpoint (e.g. data/checkpoints/dpo_seed42_taubench_v3/step_300)
#   OPENAI_API_KEY — set in environment before submitting (sbatch --export=ALL,OPENAI_API_KEY=...)
#
# No local vLLM needed — target is OpenAI API.
# Auditor runs on the single GPU; judge also via OpenAI API (or override MODELS_CONFIG).

set -euo pipefail

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/lora/dir}"
EPISODES="${EPISODES:-50}"
MAX_TURNS="${MAX_TURNS:-14}"
SEED="${SEED:-42}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_transfer_gpt4omini.yaml}"

PROBE_DIR="/shared/home/kshitij.mishra/aapo"
CONDA_BASE="/shared/home/kshitij.mishra/miniconda3"
ENV_BIN="${CONDA_BASE}/envs/aapo/bin"
HF_HOME="/shared/home/kshitij.mishra/.cache/huggingface"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set — use: sbatch --export=ALL,OPENAI_API_KEY=sk-... ...}"

cd "$PROBE_DIR"
mkdir -p slurm/logs data/eval

export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== PROBE Transfer Eval — DPO auditor vs GPT-4o-mini | Checkpoint=$CHECKPOINT | Job=$SLURM_JOB_ID | $(date) ==="

# No vLLM — auditor on GPU 0
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

echo "=== Transfer eval exit=$RC | $(date) ==="
exit $RC
