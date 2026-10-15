#!/bin/bash
#SBATCH --job-name=probe-setup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/setup_%j.out
#SBATCH --error=slurm/logs/setup_%j.err

set -euo pipefail

PROBE_DIR="/shared/home/kshitij.mishra/aapo"
CONDA_BASE="/shared/home/kshitij.mishra/miniconda3"
ENV_NAME="probe"
ENV_BIN="${CONDA_BASE}/envs/${ENV_NAME}/bin"
HF_HOME="/shared/home/kshitij.mishra/.cache/huggingface"
VENDOR_DIR="${PROBE_DIR}/vendor"

cd "$PROBE_DIR"
mkdir -p slurm/logs "${HF_HOME}" "${VENDOR_DIR}"

echo "=== PROBE Environment Setup | Job=$SLURM_JOB_ID | $(date) ==="

# ── 1. Create conda env if it doesn't exist ───────────────────────────
if [ ! -d "${CONDA_BASE}/envs/${ENV_NAME}" ]; then
    echo "[1/6] Creating conda env '${ENV_NAME}' (Python 3.11)..."
    "${CONDA_BASE}/bin/conda" create -n "${ENV_NAME}" python=3.11 -y
else
    echo "[1/6] Conda env '${ENV_NAME}' already exists — skipping create"
fi

# ── 2. Install PyTorch (CUDA 12.1) ───────────────────────────────────
echo "[2/6] Installing PyTorch 2.10.0+cu128 + torchvision (Blackwell/cc12.x support)..."
"${ENV_BIN}/pip" install --quiet \
    torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# ── 3. Install PROBE dependencies ──────────────────────────────────────
# Install vllm first; it pins tokenizers>=0.22. Then force transformers>=4.51
# which removed the <0.22 upper bound on tokenizers.
echo "[3/6] Installing vllm..."
"${ENV_BIN}/pip" install --quiet "vllm>=0.4"

echo "[3/6] Installing remaining PROBE dependencies..."
"${ENV_BIN}/pip" install --quiet \
    accelerate \
    "peft>=0.7" \
    "trl>=0.7" \
    "bitsandbytes>=0.43" \
    "sentence-transformers>=2.2" \
    "hdbscan>=0.8" \
    scikit-learn \
    datasets \
    "omegaconf>=2.3" \
    pyyaml \
    jsonlines \
    wandb \
    matplotlib \
    seaborn \
    scipy \
    "openai>=1.0" \
    safetensors \
    "liger-kernel>=0.5"

# Force transformers>=4.51 — this version lifted the tokenizers<0.22 constraint.
# Must run after all other installs so nothing downgrades it back to 4.48.
echo "[3/6c] Upgrading transformers to >=4.51 (compatible with tokenizers 0.22)..."
"${ENV_BIN}/pip" install --quiet --upgrade "transformers>=4.51.0"

# Flash Attention 2 (needs CUDA headers; may fail on some nodes — non-fatal)
echo "[3/6b] Installing flash-attn (may take a while)..."
"${ENV_BIN}/pip" install --quiet flash-attn --no-build-isolation 2>/dev/null \
    && echo "  flash-attn installed" || echo "  flash-attn skipped (non-fatal)"

# ── 4. Clone and install tau-bench ───────────────────────────────────
echo "[4/6] Setting up tau-bench..."
if [ ! -d "${VENDOR_DIR}/tau-bench" ]; then
    git clone https://github.com/sierra-research/tau-bench "${VENDOR_DIR}/tau-bench"
else
    echo "  tau-bench already cloned"
    cd "${VENDOR_DIR}/tau-bench" && git pull --quiet && cd "$PROBE_DIR"
fi
"${ENV_BIN}/pip" install --quiet -e "${VENDOR_DIR}/tau-bench"
echo "  tau-bench installed"

# ── 5. Pre-fetch sentence-transformers MiniLM (~90 MB) ───────────────
echo "[5/6] Downloading sentence-transformers/all-MiniLM-L6-v2..."
export HF_HOME="${HF_HOME}"
"${ENV_BIN}/python" -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print('  MiniLM downloaded OK')
" && echo "  MiniLM OK" || echo "  WARNING: MiniLM download failed (non-fatal — will retry at runtime)"

# ── 6. Init flights database ─────────────────────────────────────────
echo "[6/6] Initializing flights database..."
"${ENV_BIN}/python" scripts/init_db.py --seed 42
echo "  data/flights.db created"

echo ""
echo "=== Setup complete | $(date) ==="
echo ""
echo "Next steps — submit in order:"
echo "  Smoke test:  sbatch slurm/smoke_test_mbzuai.sbatch"
echo "  Training:    METHOD=dpo SEED=42 RUN_TAG=_taubench_v3 sbatch slurm/train_taubench_mbzuai.sbatch"
