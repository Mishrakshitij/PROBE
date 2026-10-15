#!/bin/bash
#SBATCH --job-name=aapo-transfer-llama
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --account=cscc-users
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-sxm4-40gb:2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/transfer_%x_%j.out
#SBATCH --error=slurm/logs/transfer_%x_%j.err

set -euo pipefail

# Cross-family transfer eval (bwc6 Q2): Qwen-trained DPO-V3 auditor vs a Llama target.
CHECKPOINT="${CHECKPOINT:-data/checkpoints/dpo_seed42_v3/step_300}"
EPISODES="${EPISODES:-50}"
MAX_TURNS="${MAX_TURNS:-8}"
SEED="${SEED:-42}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/models_transfer_llama.yaml}"
TARGET_MODEL="${TARGET_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
TARGET_PARSER="${TARGET_PARSER:-llama3_json}"
OUT_TAG="${OUT_TAG:-transfer_llama}"
# Per-server conda env + quantization overrides. The judge (Qwen-30B) stays in
# AAPO_CONDA_ENV (its vLLM 0.16 + working bitsandbytes handle the 30B). A newer
# target (e.g. Gemma-4) is served from TARGET_VLLM_ENV (haoyi2_vllm = vLLM
# 0.22.1); that env's bitsandbytes is too old for its Gemma-4 loader, so serve
# the target unquantized (TARGET_QUANT=none, bump TARGET_UTIL). Servers are
# reached over HTTP so vLLM versions may differ. VLLM_ENV kept for back-compat.
VLLM_ENV="${VLLM_ENV:-}"
JUDGE_VLLM_ENV="${JUDGE_VLLM_ENV:-}"
TARGET_VLLM_ENV="${TARGET_VLLM_ENV:-$VLLM_ENV}"
TARGET_QUANT="${TARGET_QUANT:-bitsandbytes}"
TARGET_UTIL="${TARGET_UTIL:-0.45}"

export AAPO_HOME="${AAPO_HOME:-/home/salem.lahlou/kmishra/aapo}"
export AAPO_CONDA_ENV="${AAPO_CONDA_ENV:-/nfs-stor/salem.lahlou/conda_envs/kshitij-aapo}"
export HF_HOME="${HF_HOME:-/nfs-stor/salem.lahlou/hf_cache}"

cd "$AAPO_HOME"
export PATH="$AAPO_CONDA_ENV/bin:$PATH"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="$AAPO_CONDA_ENV"

mkdir -p slurm/logs data/eval "$HF_HUB_CACHE"

echo "AAPO Transfer Eval — auditor=$CHECKPOINT vs target=$TARGET_MODEL | Job=$SLURM_JOB_ID Node=$SLURM_NODELIST | $(date)"

[ ! -f data/flights.db ] && python scripts/init_db.py --seed "$SEED"

# GPU 0 = Qwen-30B judge (fixed, same as paper). GPU 1 = Llama target + 4B auditor.
JUDGE_PID=$(bash slurm/helpers/start_vllm_generic.sh "Qwen/Qwen3-30B-A3B-Instruct-2507" 0 8001 "slurm/logs/vllm_judge_${SLURM_JOB_ID}.log" hermes 2048 0.72 "$JUDGE_VLLM_ENV" bitsandbytes)
TARGET_PID=$(bash slurm/helpers/start_vllm_generic.sh "$TARGET_MODEL" 1 8003 "slurm/logs/vllm_target_${SLURM_JOB_ID}.log" "$TARGET_PARSER" 8192 "$TARGET_UTIL" "$TARGET_VLLM_ENV" "$TARGET_QUANT")

python slurm/helpers/wait_vllm.py 8001 600 || { kill $JUDGE_PID $TARGET_PID 2>/dev/null; exit 1; }
python slurm/helpers/wait_vllm.py 8003 600 || { kill $JUDGE_PID $TARGET_PID 2>/dev/null; exit 1; }

# BASELINE=<scripted|fixed|coop|fewshot> runs a non-learning baseline auditor
# against the SAME transfer target instead of a checkpoint — the comparative
# reference point for the cross-family number (Xtwt W1). Static baselines need
# no auditor GPU; they reuse the target/judge servers from MODELS_CONFIG.
BASELINE="${BASELINE:-}"
if [ -n "$BASELINE" ]; then
    python scripts/run_rebuttal_baselines.py \
        --baseline "$BASELINE" \
        --episodes "$EPISODES" --max-turns "$MAX_TURNS" \
        --models-config "$MODELS_CONFIG" --seed "$SEED" \
        --output "data/eval/baseline_${BASELINE}_${OUT_TAG}_report.json" \
        --log-dir "data/eval/baseline_${BASELINE}_${OUT_TAG}_logs"
    RC=$?
    kill $JUDGE_PID $TARGET_PID 2>/dev/null; wait 2>/dev/null || true
    echo "DONE: transfer baseline '$BASELINE' exit=$RC | $(date)"
    exit $RC
fi

# NO_ADAPTER=1 runs the untrained base auditor against the same transfer target —
# the comparative baseline for the cross-family number (Xtwt W1).
NO_ADAPTER="${NO_ADAPTER:-0}"
PREFIX="dpo_v3_s300"
EVAL_ARGS=(--episodes "$EPISODES" --max-turns "$MAX_TURNS"
           --models-config "$MODELS_CONFIG" --auditor-gpu 1 --seed "$SEED")
if [ "$NO_ADAPTER" = "1" ]; then
    PREFIX="zeroshot_4b"
    EVAL_ARGS+=(--no-adapter --tag "${PREFIX}_${OUT_TAG}")
else
    EVAL_ARGS+=(--checkpoint "$CHECKPOINT")
fi
EVAL_ARGS+=(--output "data/eval/${PREFIX}_${OUT_TAG}_report.json"
            --log-dir "data/eval/${PREFIX}_${OUT_TAG}_logs")

python scripts/eval_checkpoint.py "${EVAL_ARGS[@]}"
RC=$?

kill $JUDGE_PID $TARGET_PID 2>/dev/null; wait 2>/dev/null || true
echo "DONE: transfer eval exit=$RC | $(date)"
exit $RC
