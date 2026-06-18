# AAPO: Auditing Agents for Policy-violation Observation

Automated auditor that trains an LLM to play the "user" role against a tool-augmented target agent (flight-booking domain), generating diverse multi-turn interactions that surface policy violations.

## Architecture

```
configs/ (YAML)  →  types.py + config.py  →  environment/  →  trajectory/
                                              auditor/         reward/
                                              inference/       training/
                                                               replay/
                                                               evaluation/
```

### Data flow (one training step)

```
CurriculumScheduler ──(K_max, turn t)──►
    ReplayBuffer ──(prefix S_t)──►
        CandidateSampler ──(G candidates)──►
            SharedPrefixRunner ──(G continuations)──►
                CompositeReward ──(scored group)──►
                    PolicyUpdater ──(update π)──►
                        ReplayBuffer + ClusterManager
```

## Inference Backend

All LLM consumers (TargetAgent, LLMJudge, ZeroShotAuditor) use the `InferenceBackend` protocol:

```python
Protocol: InferenceBackend
  .generate(messages, **kwargs) → str
  .generate_with_tools(messages, tools, **kwargs) → (str, list[ToolCall])
```

**Implementations:**
- `VLLMClient` — calls local vLLM server via OpenAI-compatible API
- `APIClient` — calls external API via litellm (optional fallback)
- Auditor uses HuggingFace transformers directly (for LoRA training)

Swapping models = editing `configs/models.yaml`. No code changes needed.

### GPU Allocation

```
TRAINING:
  GPU 7 → vLLM: Qwen3-30B-A3B-Instruct-2507 bitsandbytes 4-bit (~17GB)
           serves BOTH target agent AND judge (same model, different prompts)
           prefix caching ON for shared-prefix candidate sampling
  GPU 5 → Auditor: Qwen3-4B-Instruct-2507 QLoRA 4-bit
           ~3GB weights + ~1GB LoRA + ~1GB optim + ~5GB activations

EVALUATION:
  GPU 7 → vLLM: target (Qwen3-30B-A3B-Instruct-2507, same as training)
  GPU 3+4 → vLLM: stronger judge (Qwen3-Next-80B-A3B-Instruct, tensor parallel)
```

### Model Config (`configs/models.yaml`)

```yaml
models:
  target:
    name: "Qwen/Qwen3-30B-A3B-Instruct-2507"
    backend: "vllm"
    gpu: 7
    quantization: "bitsandbytes"
    port: 8001

  judge:
    name: "Qwen/Qwen3-30B-A3B-Instruct-2507"
    backend: "vllm"
    gpu: 7
    port: 8001  # shared server

  auditor:
    name: "Qwen/Qwen3-4B-Instruct-2507"
    backend: "transformers"
    gpu: 5
    quantization: "bitsandbytes-4bit"
    lora:
      rank: 64
      alpha: 128
      target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

## Environment

**Flight-booking domain** with 10-table SQLite DB:
- Real-world backbone: airports, airlines, routes from OpenFlights data
- Synthetic overlay: fares, schedules, edge cases at configurable rates
- Each `(seed, config)` produces a unique, reproducible DB

**8 tools**: search_flights, get_flight_details, check_availability, get_booking_details, book_flight, cancel_booking, modify_booking, hold_itinerary

**20 policies** across 4 categories (16 deterministic, 4 hybrid):
- Business (B1-B6): profitability, USD-only, restricted routes, upsell, fare disclosure, round-trip
- Safety (S1-S6): tool gating, schema adherence, no fabrication, safe defaults, confirmation, error handling
- Privacy (P1-P4): no internal pricing, no other passengers, no tool schema, PII handling
- Operational (O1-O4): max passengers, advance booking, min connection time, cancelled flights

**Target agent**: "SkyAssist" — system prompt deliberately has gaps to create exploitable policies.

## Training methods

- **GRPO**: Group-based relative policy optimization with shared-prefix candidate sampling
- **DPO**: Direct preference optimization from candidate pair rankings
- **GFlowNet**: Trajectory-balance loss for reward-proportional (mode-covering) sampling

### Reward system

```
r_k = r_viol + α·r_prog + β·r_nov + γ·r_plaus
```

- Violation reward: severity-weighted, per-turn attributed
- Progress shaping: precondition bonuses for multi-turn setup (decays over training)
- Novelty: cluster-based + inverse-frequency policy bonus (increases over training)
- Plausibility: penalizes adversarial/unnatural auditor behavior (constant)
- All components independently z-score normalized, weights scheduled

## Setup

```bash
pip install -e ".[dev]"

# For local vLLM inference:
pip install -e ".[vllm]"

# For external API fallback:
pip install -e ".[api]"
```

## Usage

### 1. Initialize database

```bash
python scripts/init_db.py --seed 42
python scripts/init_db.py --seed 123  # different seed = different DB
```

### 2. Launch inference servers

```bash
# Training profile: target+judge on GPU 7
python scripts/launch_servers.py --profile training

# Eval profile: target on GPU 7, stronger judge on GPU 3+4
python scripts/launch_servers.py --profile eval
```

### 3. Run baselines

```bash
python scripts/run_baseline.py --baseline B0 --episodes 100  # random
python scripts/run_baseline.py --baseline B2 --episodes 100  # zero-shot LLM
```

### 4. Train auditor

```bash
python scripts/train.py --method grpo --steps 1000 --seed 42
python scripts/train.py --method dpo --steps 500
python scripts/train.py --method gflownet --steps 500
```

LoRA checkpoints saved to `data/checkpoints/` every 200 steps.

### 5. Evaluate

```bash
python scripts/evaluate.py --log-dir data/logs
```

### 6. Export failure catalog

```bash
python scripts/export_catalog.py --output data/failure_catalog.json
```

## Testing

```bash
pytest tests/ -v
```

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| InferenceBackend protocol | Swap models via config — VLLMClient, APIClient, or any new backend |
| vLLM with prefix caching | Fast local inference; shared-prefix sampling gets near-free caching |
| QLoRA 4-bit auditor | Fits on 14GB GPU with room for optimizer states |
| Shared vLLM server for target+judge | Same model, different prompts; saves GPU memory |
| Real-world data backbone (OpenFlights) | Generalization: auditor learns strategies, not memorized patterns |
| Parameterized DB generator | Different seed = different DB for train/test split |
| 20 programmatic verifiers | Stable, fast reward signal; LLM judge only for 4 hybrid policies |
| Running z-score normalization | Reward components on different scales; prevents one from dominating |
| Scheduled weights (α↓, β↑) | Progress shaping matters early; novelty matters late past plateaus |
| Softplus floor on R_total | Required for GFlowNet TB loss (R > 0) |
| Shared-prefix candidate sampling | O(G) cost instead of O(G·K) for comparable rollouts |
| EMA sampling policy | More stable candidate generation than actively-training policy |
| Curriculum over K_max | Start easy (short episodes), grow to hard (long-horizon violations) |

## Project structure

```
aapo/
├── configs/
│   ├── environment.yaml   # DB params, target agent settings
│   ├── models.yaml        # Model names, GPUs, quantization, LoRA, ports
│   ├── policies.yaml      # 20 policies with IDs, categories, severity
│   ├── training.yaml      # GRPO/DPO/GFlowNet hyperparams, curriculum
│   ├── reward.yaml        # Weights α/β/γ, normalization, thresholds
│   └── eval.yaml          # Eval episode count, seeds, replay config
│
├── src/aapo/
│   ├── types.py           # Shared frozen dataclasses
│   ├── config.py          # Typed config loader (OmegaConf)
│   ├── inference/         # VLLMClient, APIClient, server lifecycle, model loader
│   ├── environment/       # FlightDB, 8 tools, 20 policies, SkyAssist target agent
│   ├── auditor/           # AuditorPolicy (QLoRA), CandidateSampler, EMAPolicy
│   ├── trajectory/        # EpisodeRunner, SharedPrefixRunner
│   ├── reward/            # 20 verifiers, LLM judge, attribution, shaping, novelty, composite
│   ├── training/          # GRPO, DPO, GFlowNet, curriculum, training loop
│   ├── replay/            # Prioritized buffer, embedder, HDBSCAN clustering, dedup, catalog
│   └── evaluation/        # Coverage, diversity, yield, efficiency metrics + reporting
│
├── scripts/
│   ├── init_db.py         # Create + populate flight DB
│   ├── launch_servers.py  # Start/stop vLLM servers by profile
│   ├── train.py           # Main training entry point
│   ├── run_baseline.py    # Run baselines B0/B2
│   ├── evaluate.py        # Evaluation suite
│   └── export_catalog.py  # Generate failure catalog
│
├── tests/                 # Unit tests for all layers
└── data/                  # Generated (gitignored)
```

## Changes log

### vLLM Migration (2026-02-23)

**New files:**
- `configs/models.yaml` — central model config (names, GPUs, quantization, ports, LoRA)
- `src/aapo/inference/__init__.py` — inference package
- `src/aapo/inference/vllm_client.py` — `VLLMClient`, `APIClient`, `InferenceBackend` protocol
- `src/aapo/inference/server.py` — vLLM server launch/wait/shutdown
- `src/aapo/inference/model_loader.py` — factory: role → backend from models.yaml
- `scripts/launch_servers.py` — CLI for training/eval server profiles

**Modified files:**
- `src/aapo/environment/target_agent.py` — accepts `InferenceBackend` client instead of litellm
- `src/aapo/reward/judge.py` — accepts `InferenceBackend` client instead of litellm
- `src/aapo/auditor/model.py` — added QLoRA 4-bit support, `save_lora()`/`load_lora()`
- `src/aapo/environment/__init__.py` — lazy import for TargetAgent
- `src/aapo/config.py` — removed hardcoded model from TargetAgentConfig
- `configs/environment.yaml` — removed model field, references models.yaml
- `pyproject.toml` — added openai, bitsandbytes; moved litellm to optional `[api]`, vllm to `[vllm]`
- `scripts/train.py` — loads backends from models.yaml, QLoRA auditor, LoRA checkpointing
- `scripts/run_baseline.py` — uses InferenceBackend instead of litellm

### Section F: Verification (2026-02-25)

**Bug fixes during smoke testing:**
- `src/aapo/inference/server.py` — added `--enable-auto-tool-choice --tool-call-parser hermes` for tool calling; bumped default `gpu_memory_utilization` to 0.55
- `src/aapo/inference/vllm_client.py` — added max_tokens halving retry on context overflow
- `src/aapo/auditor/sampler.py` — added missing `generate()` method (delegates to `sampling_policy`)
- `src/aapo/auditor/ema.py` — added `is_floating_point()` guard to skip quantized uint8 params in EMA update
- `src/aapo/environment/logger.py` — added `update_episode()` to persist violations/rewards after scoring
- `src/aapo/replay/cluster.py` — auto-lower HDBSCAN `min_cluster_size` for small datasets
- `src/aapo/replay/catalog.py` — include unclustered episodes with violations as singleton clusters
- `scripts/run_baseline.py` — call `logger.update_episode()` after scoring to persist violations
- `scripts/export_catalog.py` — fixed import path for `load_episodes_from_logs`
- `configs/models.yaml` — `max_model_len: 8192`, `max_tokens: 512`, auditor on GPU 5

**Smoke test results:**
- B0 baseline: 0 violations (expected — random dates don't match DB)
- B2 baseline (20 ep): 124 violations, 9/20 policies, 45% coverage
- Training smoke (5 GRPO steps): completed, LoRA saved
- Failure catalog: 19 clusters exported

**New files:**
- `scripts/smoke_test.py` — 5-turn scripted integration test against live vLLM target
