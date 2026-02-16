# AAPO: Auditing Agents for Policy-violation Observation

Automated auditor that trains an LLM to play the "user" role against a tool-augmented target agent (flight-booking domain), generating diverse multi-turn interactions that surface policy violations.

## Architecture

```
configs/ (YAML)  →  types.py + config.py  →  environment/  →  trajectory/
                                              auditor/         reward/
                                                               training/
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
```

## Usage

### 1. Initialize database

```bash
python scripts/init_db.py --seed 42
python scripts/init_db.py --seed 123  # different seed = different DB
```

### 2. Run baselines

```bash
python scripts/run_baseline.py --baseline B0 --episodes 100  # random
python scripts/run_baseline.py --baseline B2 --episodes 100  # zero-shot LLM
```

### 3. Train auditor

```bash
python scripts/train.py --method grpo --steps 1000 --seed 42
python scripts/train.py --method dpo --steps 500
python scripts/train.py --method gflownet --steps 500
```

### 4. Evaluate

```bash
python scripts/evaluate.py --log-dir data/logs
```

### 5. Export failure catalog

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
├── configs/           # YAML configs for environment, policies, training, reward, eval
├── src/aapo/
│   ├── types.py       # Shared frozen dataclasses
│   ├── config.py      # Typed config loader (OmegaConf)
│   ├── environment/   # FlightDB, 8 tools, 20 policies, SkyAssist target agent
│   ├── auditor/       # AuditorPolicy, CandidateSampler, EMAPolicy
│   ├── trajectory/    # EpisodeRunner, SharedPrefixRunner
│   ├── reward/        # 20 verifiers, LLM judge, attribution, shaping, novelty, composite
│   ├── training/      # GRPO, DPO, GFlowNet, curriculum, training loop
│   ├── replay/        # Prioritized buffer, embedder, HDBSCAN clustering, dedup, catalog
│   └── evaluation/    # Coverage, diversity, yield, efficiency metrics + reporting
├── scripts/           # init_db, train, run_baseline, evaluate, export_catalog
├── tests/             # Unit tests for all layers
└── data/              # Generated (gitignored)
```
