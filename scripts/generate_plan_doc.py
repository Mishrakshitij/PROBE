#!/usr/bin/env python3
"""Generate AAPO_Experiment_Plan.docx — comprehensive plan + codebase docs.

This document is intended for a teammate to replicate the AAPO project on a
second cluster, running planned experiments in parallel.
"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


doc = Document()

# Styles
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

for level in range(1, 4):
    h = doc.styles[f'Heading {level}']
    h.font.color.rgb = RGBColor(0x1a, 0x47, 0x7a)


def code(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Cm(0.5)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p


def table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = h
        for p in c.paragraphs:
            p.runs[0].bold = True
            p.runs[0].font.size = Pt(10)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            c = t.rows[r_idx + 1].cells[c_idx]
            c.text = str(val)
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
    return t


def bullet(text):
    doc.add_paragraph(text, style='List Bullet')


# =============================================================
# COVER PAGE
# =============================================================
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\n\n\n\nAAPO')
run.font.size = Pt(36); run.bold = True
run.font.color.rgb = RGBColor(0x1a, 0x47, 0x7a)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Experimental Plan & Codebase Onboarding')
run.font.size = Pt(18)
run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\n\nFor parallel execution across compute clusters')
run.font.size = Pt(13)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('\n\nNeurIPS 2026 Submission Track')
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

doc.add_page_break()

# =============================================================
# TOC
# =============================================================
doc.add_heading('Table of Contents', level=1)
for item in [
    '1. Project Goals and Current Status',
    '2. Repository Setup (any cluster)',
    '3. Codebase Walkthrough',
    '4. Training Pipeline Architecture',
    '5. Experiments Completed (V1, V2, V3)',
    '6. Supervisor Feedback Summary',
    '7. Planned Experiments (Priority Ordered)',
    '8. Cluster Configurations (Long, CSCC, External)',
    '9. How to Parallelize Across Clusters',
    '10. Code Modifications Needed Before Each Experiment',
    '11. Data Sync and Result Aggregation',
    '12. Risk Register and Common Failures',
    '13. Submission Timeline',
]:
    p = doc.add_paragraph(item)
    p.paragraph_format.space_after = Pt(2)

doc.add_page_break()

# =============================================================
# 1. PROJECT GOALS
# =============================================================
doc.add_heading('1. Project Goals and Current Status', level=1)

doc.add_heading('What We Are Building', level=2)
doc.add_paragraph(
    'AAPO trains a small adversarial language model (Qwen3-4B with QLoRA) to play '
    'the role of a customer and find policy violations in a tool-augmented LLM agent '
    '(Qwen3-30B target). The auditor is optimized via reinforcement learning with three '
    'algorithms (GRPO, DPO, GFlowNet) and a curriculum that gradually extends conversation '
    'length. The target agent operates a flight-booking environment with 20 operational '
    'policies and 8 booking tools.'
)

doc.add_heading('Headline Numbers (V2)', level=2)
table(
    ['Method', 'Episodes', 'Coverage', 'Yield/100', 'Multi-policy', 'Entropy'],
    [
        ['GRPO V2 step 50', 50, '60% (12/20)', 1604, '94%', 3.01],
        ['DPO V2 step 50', 50, '60% (12/20)', 1528, '90%', 3.08],
        ['GFlowNet V2 step 50', 50, '60% (12/20)', 1700, '94%', 3.12],
        ['GFlowNet V2 step 250', 50, '55% (11/20)', 1614, '94%', 2.96],
        ['GRPO V3 step 50', 50, '55% (11/20)', 1028, '82%', 3.31],
    ]
)

doc.add_heading('What We Have Right Now', level=2)
for item in [
    '12 training runs, 2,617 step-level metric records (76 fields per step)',
    '8 evaluation reports, 1,068 logged episodes',
    '32 LoRA checkpoints across V1, V2, V3',
    'Complete RL pipeline working end-to-end with no crashes',
    'Reward engineering progression V1 to V3 (system prompt, scenarios, plausibility, deduplication, anti-fabrication)',
]:
    bullet(item)

doc.add_heading('Pending', level=2)
for item in [
    'V3 evaluations (5+ jobs running on long/cscc partitions)',
    'B0 random and B2 zero-shot baselines (200 episodes each)',
    'Ablation experiments isolating which V3 fix matters',
    'Multi-seed runs (currently only seed 42)',
    'Transfer to a different target model (GPT-4o-mini, Llama)',
    'Decision on environment switch to tau-bench',
]:
    bullet(item)

doc.add_page_break()

# =============================================================
# 2. REPOSITORY SETUP
# =============================================================
doc.add_heading('2. Repository Setup (any cluster)', level=1)

doc.add_heading('Clone the Repo', level=2)
code('git clone https://github.com/Mishrakshitij/AAPO.git\n'
     'cd AAPO\n'
     'git checkout main')

doc.add_heading('Conda Environment', level=2)
doc.add_paragraph('Python 3.11. The conda environment is at:')
code('/nfs-stor/salem.lahlou/conda_envs/kshitij-aapo')
doc.add_paragraph('On a new cluster, recreate it with:')
code('conda create -n aapo python=3.11 -y\n'
     'conda activate aapo\n'
     'pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121\n'
     'pip install transformers accelerate peft bitsandbytes vllm\n'
     'pip install datasets sentence-transformers hdbscan scikit-learn\n'
     'pip install omegaconf pyyaml jsonlines wandb\n'
     'pip install matplotlib seaborn scipy\n'
     'pip install liger-kernel flash-attn --no-build-isolation')

doc.add_heading('Hugging Face Cache', level=2)
doc.add_paragraph('Pre-download the models to a shared cache:')
code('export HF_HOME=/path/to/hf_cache\n'
     'export HF_HUB_CACHE=$HF_HOME/hub\n'
     'huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507\n'
     'huggingface-cli download Qwen/Qwen3-4B-Instruct-2507\n'
     'huggingface-cli download sentence-transformers/all-MiniLM-L6-v2')

doc.add_heading('Initialize Database', level=2)
code('python scripts/init_db.py --seed 42')
doc.add_paragraph('Generates SQLite DB at data/flights.db with 35 airports, 120 routes, 643 flights.')

doc.add_heading('Verify Installation', level=2)
code('python scripts/smoke_test.py')
doc.add_paragraph('Should run a few episodes with a random auditor and confirm everything works.')

doc.add_page_break()

# =============================================================
# 3. CODEBASE WALKTHROUGH
# =============================================================
doc.add_heading('3. Codebase Walkthrough', level=1)

doc.add_heading('Top-level Layout', level=2)
code(
    'aapo/\n'
    '  configs/                       YAML configuration files\n'
    '    models_slurm.yaml             Model paths, GPUs, ports for SLURM\n'
    '    reward.yaml                   Reward weights and schedule\n'
    '    training.yaml                 Per-method hyperparameters\n'
    '    policies.yaml                 20 policy definitions\n'
    '\n'
    '  scripts/                       Entry points\n'
    '    train.py                      Main training entry\n'
    '    evaluate.py                   Analyze eval logs\n'
    '    eval_checkpoint.py            Run new eval on a checkpoint\n'
    '    init_db.py                    Initialize flight DB\n'
    '    run_baseline.py               B0 random and B2 zero-shot baselines\n'
    '    plot_results.py               Generate paper figures\n'
    '\n'
    '  src/aapo/                      Main Python package\n'
    '    types.py                      Core dataclasses\n'
    '    config.py                     Config loader\n'
    '    auditor/                      Auditor model + scenarios + EMA\n'
    '    environment/                  Flight DB + tools + target + policies\n'
    '    inference/                    vLLM client wrapper\n'
    '    reward/                       Composite reward + verifiers + judge\n'
    '    training/                     Loop + GRPO/DPO/GFlowNet + curriculum\n'
    '    trajectory/                   Episode + prefix runners\n'
    '    replay/                       Buffer + clustering\n'
    '    evaluation/                   Metrics + report generation\n'
    '\n'
    '  slurm/                         SLURM job scripts\n'
    '    train_single.sbatch           Per-method training (long, 72h)\n'
    '    train_single_cscc.sbatch      Per-method training (cscc, 24h)\n'
    '    eval_single.sbatch            Per-checkpoint eval (long)\n'
    '    eval_single_cscc.sbatch       Per-checkpoint eval (cscc)\n'
    '    helpers/start_vllm.sh         Launch vLLM server\n'
    '    helpers/wait_vllm.py          Poll until vLLM ready\n'
    '\n'
    '  data/                          Generated artifacts (gitignored)\n'
    '    flights.db                    SQLite database\n'
    '    logs/{run_name}/              Training metrics JSONL\n'
    '    eval/{checkpoint_name}/       Evaluation reports + episode logs\n'
    '    checkpoints/{run_name}/       LoRA adapters\n'
)

doc.add_heading('Critical Source Files', level=2)
table(
    ['Path', 'Lines', 'Purpose'],
    [
        ['src/aapo/auditor/model.py', 320, 'AuditorPolicy: Qwen3-4B + QLoRA wrapper'],
        ['src/aapo/auditor/scenarios.py', 280, 'ScenarioPool: 27 scenarios + DB-aware generation'],
        ['src/aapo/environment/target_agent.py', 155, 'TargetAgent: SkyAssist with 8 tools'],
        ['src/aapo/environment/policies.py', 106, '20 policies (B1-O4)'],
        ['src/aapo/reward/composite.py', 165, 'CompositeReward with weight schedule, normalization'],
        ['src/aapo/reward/attribution.py', 85, 'Verifier dispatch + diminishing returns'],
        ['src/aapo/reward/verifiers.py', 546, '20 programmatic policy verifiers'],
        ['src/aapo/reward/plausibility.py', 130, 'Plausibility scoring (V3 with anti-fab)'],
        ['src/aapo/reward/shaping.py', 145, 'Progress shaping (V3 3-tier)'],
        ['src/aapo/training/loop.py', 365, 'Main training orchestrator'],
        ['src/aapo/training/grpo.py', 152, 'GRPO updater'],
        ['src/aapo/training/dpo.py', 138, 'DPO updater'],
        ['src/aapo/training/gflownet.py', 122, 'GFlowNet TB updater'],
        ['src/aapo/training/curriculum.py', 95, 'k_max stages, advancement logic'],
        ['src/aapo/trajectory/episode_runner.py', 130, 'Full episode + loop detection'],
        ['src/aapo/trajectory/prefix_runner.py', 100, 'Shared-prefix candidate sampling'],
    ]
)

doc.add_page_break()

# =============================================================
# 4. TRAINING PIPELINE ARCHITECTURE
# =============================================================
doc.add_heading('4. Training Pipeline Architecture', level=1)

doc.add_heading('One Training Step (V3)', level=2)
for i, (title, desc) in enumerate([
    ('Sample Scenario',
     'ScenarioPool.from_db() draws a customer scenario. Inject into auditor system prompt and EMA prompt.'),
    ('Curriculum',
     'CurriculumScheduler.sample_turn_index() picks a branching point t in current k_max stage [4,8,12,14].'),
    ('Get Prefix',
     'ReplayBuffer.sample_prefix(t) returns a 2t-turn conversation prefix, or runs a fresh episode if needed.'),
    ('Generate Candidates',
     'Auditor.generate_candidates(prefix, n=4) produces G=4 alternative user messages from same prefix.'),
    ('Get Continuations',
     'SharedPrefixRunner runs 3 target turns per candidate, in parallel via ThreadPoolExecutor + vLLM.'),
    ('Score Episode',
     'CompositeReward computes r_viol + alpha*r_prog + beta*r_nov + gamma*r_plaus, with z-score norm and softplus floor.'),
    ('Apply Diminishing Returns',
     'Per-policy violation weights decay 1.0 / 0.5 / 0.0 across episode (V3 fix).'),
    ('Update Auditor',
     'GRPO/DPO/GFlowNet computes gradient and updates LoRA weights. clip_grad_norm 1.0.'),
    ('Update EMA',
     'theta_ema = 0.995 * theta_ema + 0.005 * theta_train.'),
    ('Curriculum Advance',
     'If avg violation_yield >= 15 percent for the step, curriculum advances to next k_max stage.'),
    ('Log Metrics',
     'TrainingLogger writes JSONL row with 76 fields including loss, yield, completion length, all reward components.'),
    ('Checkpoint',
     'Every 25 steps: save LoRA adapter to data/checkpoints/{run_name}/step_{N}/.'),
]):
    doc.add_heading(f'Step {i+1}: {title}', level=3)
    doc.add_paragraph(desc)

doc.add_heading('Key Design Decisions', level=2)
for item in [
    'Shared-prefix sampling (not full rollouts): G=4 candidates from one prefix, 3-turn target continuation each. Enables GRPO group baselines and DPO winner-loser pairs from the same prefix.',
    'Z-score normalization: each reward component (violation, progress, novelty, plausibility) is independently z-normalized using Welford running statistics. Prevents component imbalance.',
    'Softplus floor: total reward total = log(1 + exp(total)) ensures R > 0 for GFlowNet trajectory balance.',
    'EMA policy: stable sampling distribution, decay 0.995. Used for DPO reference and GFlowNet sampling.',
    'Gradient checkpointing during differentiable forward passes: reduces memory from ~10GB to ~2GB. Required to fit on A100-40GB.',
    'Threading lock around tokenizer: HuggingFace fast tokenizer is not thread-safe; ThreadPoolExecutor needs sync.',
]:
    bullet(item)

doc.add_page_break()

# =============================================================
# 5. EXPERIMENTS COMPLETED
# =============================================================
doc.add_heading('5. Experiments Completed (V1, V2, V3)', level=1)

doc.add_heading('V1 Baseline', level=2)
doc.add_paragraph(
    'Generic system prompt ("you are a realistic airline customer"), no scenarios, '
    'continuation_turns=1, plausibility weight gamma=0.1.'
)
doc.add_heading('V1 Failures Discovered', level=3)
table(
    ['Issue', 'Symptom', 'Root Cause'],
    [
        ['Violations not populated', 'yield=0.000 every step', 'CompositeReward bug'],
        ['CUDA OOM at k=12', 'Crash at stage 2', 'Batched G=4 forward pass'],
        ['Tokenizer "Already borrowed"', 'Random crashes', 'Thread-unsafe tokenizer'],
        ['Use cache lost', 'Generation hangs hours', 'gradient_checkpointing_disable'],
        ['Role confusion 58%', 'Auditor acts as assistant', 'No explicit role constraint'],
        ['Coverage 9/20', 'Limited policy discovery', 'continuation_turns=1 cap'],
    ]
)

doc.add_heading('V2 Improvements', level=2)
for item in [
    'System prompt rewritten: "You are NOT the assistant. 1-3 sentences max."',
    '27 randomized customer scenarios injected per episode',
    'Plausibility scorer extended: assistant-language detection, markdown penalty, echo detection',
    '3-tier progress shaping: basic slots / tool milestones / deep flow (booking, cancellation, hold)',
    'continuation_turns increased from 1 to 3',
]:
    bullet(item)
doc.add_paragraph(
    'Result: 60% policy coverage (12/20), 11% role confusion, 60-word average messages, '
    'all methods reach final curriculum stage k=14 without crashes.'
)
doc.add_paragraph(
    'Remaining issue: B4 / S5 / B6 account for 82-85% of violations due to per-turn inflation.'
)

doc.add_heading('V3 Reward Fixes', level=2)
for item in [
    'Per-policy diminishing returns: 1st violation = 100%, 2nd = 50%, 3rd+ = 0',
    'Anti-fabrication penalty: -0.6 for "I have found", "I will book", etc.',
    'Gamma raised from 0.1 to 0.35 (plausibility competes with violation reward)',
    'DB-aware scenarios: ScenarioPool.from_db() generates 20 scenarios from real routes',
    'Loop detection: early stop if last 3 user messages share >50% word overlap',
]:
    bullet(item)
doc.add_paragraph(
    'Status: training complete (DPO 300 steps, GRPO 225 steps, GFlowNet 175 steps). '
    'DPO V3 shows strongest signal: loss 0.69 -> 0.59, accuracy 1.0, margins to 0.50. '
    'V3 evaluation in progress.'
)

doc.add_heading('Open Questions', level=2)
for item in [
    'Does V3 actually improve over V2 in evaluation (not just training)? Pending eval results.',
    'Is the "reward dominates algorithm" claim valid? V1->V2 changed 5 things at once; need ablation isolating each.',
    'Are GRPO gradient norms (21-38) anomalous vs DPO (0.4-1.6)? Possibly indicates KL term not being applied.',
    'Does the auditor learn transferable strategies? Currently only tested against one target (Qwen3-30B).',
]:
    bullet(item)

doc.add_page_break()

# =============================================================
# 6. SUPERVISOR FEEDBACK
# =============================================================
doc.add_heading('6. Supervisor Feedback Summary', level=1)

doc.add_paragraph(
    'Feedback received in late development cycle. The seven items below are incorporated '
    'into the experiment plan in Section 7.'
)

table(
    ['Issue', 'Severity', 'Action'],
    [
        ['Use tau-bench env instead of SkyAssist',
         'High',
         'Decision pending; switch likely improves credibility'],
        ['Missing citation: Lee et al. ICLR 2025 (GFlowNet for red-teaming)',
         'High',
         'Add to Sec 2 + carve out our delta (multi-turn + tool-agent)'],
        ['V1->V2->V3 changes are confounded',
         'High',
         'Run factorial ablation: 2 algos x 2 reward versions minimum'],
        ['GRPO grad norm 21-38 vs DPO 0.4-1.6 looks off',
         'Medium',
         'Audit GRPO KL term; kl_loss=0 in logs is suspicious'],
        ['Single target model -> learning claim weak',
         'Medium',
         'Run trained auditor against GPT-4o-mini or Llama'],
        ['Single seed cross-algo comparison',
         'Medium',
         'Add seeds 123 and 456 for main cells'],
        ['Paper oscillates: system vs reward>algorithm thesis',
         'Low',
         'Reframe: primary thesis = learned auditor; reward eng = methodology'],
    ]
)

doc.add_page_break()

# =============================================================
# 7. PLANNED EXPERIMENTS
# =============================================================
doc.add_heading('7. Planned Experiments (Priority Ordered)', level=1)

doc.add_heading('Priority 1: Validation Experiments (3-5 days)', level=2)

doc.add_heading('Exp A: Complete V3 Evaluations', level=3)
doc.add_paragraph(
    'Status: 7 jobs queued/running. Need full V3 picture for all 3 methods at multiple checkpoints.'
)
code('# On long partition\n'
     'EXCLUDE="gpu-13,gpu-57,gpu-11,gpu-14,gpu-15,gpu-21"\n'
     'for ckpt in data/checkpoints/{grpo,dpo,gflownet}_seed42_v3/step_{50,150,225,300}; do\n'
     '  [ -d "$ckpt" ] && CHECKPOINT=$ckpt EPISODES=50 sbatch \\\n'
     '    --exclude=$EXCLUDE slurm/eval_single.sbatch\n'
     'done')
doc.add_paragraph('Decision gate: if V3 shows clear gains over V2, proceed with reward ablation.')

doc.add_heading('Exp B: GRPO KL Audit', level=3)
doc.add_paragraph(
    'Read src/aapo/training/grpo.py. Verify the KL term against ref policy is being computed. '
    'kl_loss=0.0000 in current logs is suspicious. Likely fixes: enforce ref_policy is passed '
    'in update(), confirm kl_coeff is applied. Re-run a short GRPO V3 (50 steps) with kl_coeff=0.1 '
    'and lr=5e-6 (matching DPO).'
)

doc.add_heading('Exp C: B0 and B2 Baselines (200 episodes each)', level=3)
code('python scripts/run_baseline.py --baseline B0 --episodes 200\n'
     'python scripts/run_baseline.py --baseline B2 --episodes 200')
doc.add_paragraph('Required for main results table. Both baselines exist; just need to run.')

doc.add_heading('Priority 2: Ablations (5-7 days)', level=2)

doc.add_heading('Exp D: Reward Ablation (most important for "reward > algo" claim)', level=3)
doc.add_paragraph('Configure 2x2 factorial:')
table(
    ['Cell', 'Method', 'Reward', 'Purpose'],
    [
        ['(GRPO, V2)', 'GRPO', 'V2 + no V3 fixes', 'Baseline'],
        ['(GRPO, V3-dedup)', 'GRPO', 'V2 + diminishing returns only', 'Isolate dedup'],
        ['(DPO, V2)', 'DPO', 'V2 + no V3 fixes', 'Baseline'],
        ['(DPO, V3-dedup)', 'DPO', 'V2 + diminishing returns only', 'Isolate dedup'],
    ]
)
doc.add_paragraph(
    'How to enable diminishing returns alone: revert V3 changes to plausibility.py, '
    'gamma in reward.yaml, but keep TurnRewardAttributor.violation_reward() with prior_counts.'
)

doc.add_heading('Exp E: Algorithm x Reward Full 3x2 Factorial', level=3)
doc.add_paragraph('If Exp D shows dedup matters, extend to all 3 algos.')
doc.add_paragraph(
    'Total: 6 cells. Each cell = 1 training run + 1 eval. '
    'On a 2-cluster setup with 3 simultaneous training jobs each, this completes in ~5 days.'
)

doc.add_heading('Exp F: GFlowNet with V1-style Reward (sanity check)', level=3)
doc.add_paragraph(
    'Train GFlowNet with V1 reward (gamma=0.1, no scenarios, no diminishing returns). '
    'Tests whether TB loss compensates for bad reward design. Expected outcome: TB just matches '
    'the broken distribution; coverage stays low. This is the strongest single ablation against '
    'the "GFlowNet alone helps" hypothesis.'
)

doc.add_heading('Priority 3: Generalization (3-5 days, parallelizable)', level=2)

doc.add_heading('Exp G: Transfer to GPT-4o-mini', level=3)
doc.add_paragraph(
    'Run trained DPO V3 step 300 against GPT-4o-mini as target. Requires modifying '
    'configs/models.yaml to use OpenAI API backend. No GPU needed for the target side. '
    'Eval 50 episodes. Expected: lower coverage but nonzero, demonstrating the auditor '
    'learned generalizable strategies.'
)

doc.add_heading('Exp H: Transfer to Llama-3.1-8B-Instruct', level=3)
doc.add_paragraph(
    'Same as Exp G but with a local Llama variant. Requires 1 A100-40GB for the Llama vLLM server.'
)

doc.add_heading('Exp I: Three Seeds on Main Cells', level=3)
doc.add_paragraph(
    'Run all 3 algorithms with seed in {42, 123, 456} on V3 reward. Total 9 runs. '
    'Use this for the main results table; report mean +/- std deviation.'
)

doc.add_heading('Priority 4: tau-bench Migration (5-7 days)', level=2)

doc.add_heading('Exp J: tau-bench Integration', level=3)
doc.add_paragraph(
    'Critical for NeurIPS credibility. tau-bench provides the airline + tool + policy + LLM-user '
    'setup. Our auditor replaces their scripted user. Approximate plan:'
)
for item in [
    'Clone tau-bench repo. Read their environment definitions.',
    'Replace SkyAssist (target_agent.py) with tau-bench airline target',
    'Replace 20 SkyAssist policies with tau-bench airline policies (similar but expert-curated)',
    'Replace 8 SkyAssist tools with tau-bench airline tools',
    'Adapt verifiers to tau-bench format (most should map directly)',
    'Re-run V2 + V3 training on tau-bench env',
    'Compare against their leaderboard for direct comparability',
]:
    bullet(item)

doc.add_page_break()

# =============================================================
# 8. CLUSTER CONFIGURATIONS
# =============================================================
doc.add_heading('8. Cluster Configurations', level=1)

doc.add_heading('MBZUAI Long Partition (current primary)', level=2)
table(
    ['Setting', 'Value'],
    [
        ['Partition', 'long'],
        ['Time limit', '72 hours'],
        ['GPUs per job', '2x A100-SXM4-40GB'],
        ['Sbatch file', 'slurm/train_single.sbatch'],
        ['Job submit limit', '8 concurrent'],
        ['Bad nodes (avoid)', 'gpu-13, gpu-57 (vLLM startup failures)'],
        ['Account', 'default'],
    ]
)

doc.add_heading('MBZUAI CSCC Partition', level=2)
table(
    ['Setting', 'Value'],
    [
        ['Partition', 'cscc-gpu-p'],
        ['Time limit', '24 hours'],
        ['GPUs per job', '2x A100-SXM4-40GB'],
        ['Sbatch file', 'slurm/train_single_cscc.sbatch'],
        ['Account', 'cscc-users'],
        ['QOS', 'cscc-gpu-qos'],
        ['Job submit limit', 'separate from long; ~8 concurrent'],
    ]
)
doc.add_paragraph(
    'Both partitions accept the same training scripts; only the SBATCH headers differ. '
    'Eval jobs are well-suited to cscc since they typically finish in <24h.'
)

doc.add_heading('Generic Cluster Setup', level=2)
doc.add_paragraph(
    'For any other cluster, the requirements are:'
)
for item in [
    '2x A100-40GB or larger per training job (auditor + target vLLM)',
    'NFS or shared filesystem accessible from all nodes',
    'SLURM or compatible job scheduler',
    'Conda environment as specified in Section 2',
    'Hugging Face cache pre-populated with Qwen3-30B-A3B-Instruct-2507 and Qwen3-4B-Instruct-2507',
    'Python 3.11+ with PyTorch CUDA-enabled',
    'CUDA 12.1 or compatible',
]:
    bullet(item)

doc.add_paragraph('Create a new sbatch file based on slurm/train_single.sbatch with these adjustments:')
code('#!/bin/bash\n'
     '#SBATCH --partition=YOUR_PARTITION\n'
     '#SBATCH --account=YOUR_ACCOUNT\n'
     '#SBATCH --time=72:00:00\n'
     '#SBATCH --gres=gpu:a100:2          # adjust GPU spec\n'
     '#SBATCH --mem=64G\n'
     '#SBATCH --cpus-per-task=32\n'
     '\n'
     'cd /your/path/to/aapo\n'
     'export PATH="/your/conda/bin:$PATH"\n'
     'export HF_HOME=/your/hf_cache\n'
     'export PYTHONUNBUFFERED=1\n'
     '\n'
     '# Rest of script identical to train_single.sbatch')

doc.add_page_break()

# =============================================================
# 9. PARALLELIZATION
# =============================================================
doc.add_heading('9. How to Parallelize Across Clusters', level=1)

doc.add_heading('Division of Labor', level=2)
doc.add_paragraph(
    'With 2 clusters available, split experiments by priority:'
)
table(
    ['Cluster', 'Workload'],
    [
        ['MBZUAI Long', 'Long-running training (72h jobs): ablation cells, 3-seed runs'],
        ['MBZUAI CSCC', '24h evaluations and short ablations'],
        ['External cluster A', 'tau-bench migration training (independent codebase fork)'],
        ['External cluster B', 'Transfer experiments (target = Llama / GPT-4o-mini)'],
    ]
)

doc.add_heading('Avoiding Conflicts', level=2)
for item in [
    'Use distinct RUN_TAG values per cluster: _long, _cscc, _ext_a etc',
    'Each cluster writes to its own data/logs/{run_name} subdirectory',
    'Periodically rsync data/eval and data/logs to a central location for analysis',
    'Use git branches per experiment: exp-ablation-dedup, exp-tau-bench, exp-transfer',
    'Keep configs/models_slurm.yaml synchronized via git; cluster-specific overrides go to configs/models_{cluster}.yaml',
]:
    bullet(item)

doc.add_heading('Sample Parallel Plan (3 weeks, 2 clusters)', level=2)
table(
    ['Week', 'Long Cluster', 'CSCC Cluster', 'External Cluster'],
    [
        ['1', 'V3 evals + dedup ablation training', 'B0/B2 baselines + GRPO KL audit', 'tau-bench setup + first run'],
        ['2', 'Full 3x2 factorial', 'Eval all factorial cells', 'tau-bench V3 training'],
        ['3', 'Three seeds for main cells', 'Transfer experiments + writeup figures', 'tau-bench eval + comparison'],
    ]
)

doc.add_heading('Submission Commands (template)', level=2)
code('# Training on any cluster\n'
     'METHOD=dpo SEED=42 RUN_TAG=_v3_seed42 \\\n'
     '  sbatch --job-name=aapo-dpo-v3-s42 slurm/train_single.sbatch\n'
     '\n'
     '# Evaluation\n'
     'CHECKPOINT=data/checkpoints/dpo_seed42_v3_seed42/step_300 \\\n'
     '  EPISODES=50 sbatch slurm/eval_single.sbatch\n'
     '\n'
     '# Multi-seed batch submission\n'
     'for seed in 42 123 456; do\n'
     '  for method in grpo dpo gflownet; do\n'
     '    METHOD=$method SEED=$seed RUN_TAG=_v3_s$seed \\\n'
     '      sbatch --job-name=aapo-$method-v3-s$seed slurm/train_single.sbatch\n'
     '  done\n'
     'done')

doc.add_page_break()

# =============================================================
# 10. CODE MODIFICATIONS NEEDED
# =============================================================
doc.add_heading('10. Code Modifications Needed Before Each Experiment', level=1)

doc.add_heading('Exp B (GRPO KL Audit)', level=2)
doc.add_paragraph('File: src/aapo/training/grpo.py')
for item in [
    'Verify update() receives ref_policy parameter and passes it to per-candidate KL computation',
    'Confirm kl_coeff * kl_loss is added to total loss (currently kl_loss=0.0000)',
    'Try kl_coeff=0.1 (was 0.05) and lr=5e-6 (was 1e-5)',
    'Add unit test asserting kl_loss > 0 when ref policy differs from current',
]:
    bullet(item)

doc.add_heading('Exp D (Dedup-only Ablation)', level=2)
doc.add_paragraph('Need a way to disable individual V3 fixes. Add config flags:')
code('# configs/reward_v2_dedup.yaml — only diminishing returns, no other V3 fixes\n'
     'reward:\n'
     '  alpha_progress: 0.3\n'
     '  beta_novelty: 0.2\n'
     '  gamma_plausibility: 0.1     # V2 value\n'
     '  diminishing_returns: true   # V3 fix kept\n'
     '  anti_fabrication: false     # V3 fix disabled\n'
     '  loop_detection: false       # V3 fix disabled\n'
     '\n'
     'reward_schedule:\n'
     '  alpha_start: 0.5\n'
     '  alpha_end: 0.1\n'
     '  beta_start: 0.1\n'
     '  beta_end: 0.4\n'
     '  gamma: 0.1\n'
     '  warmup_steps: 100')
doc.add_paragraph('Wire flags into the relevant scorers:')
for item in [
    'plausibility.py: gate FABRICATION_PATTERNS check on anti_fabrication flag',
    'episode_runner.py: gate loop detection on loop_detection flag',
    'composite.py: gate diminishing returns on diminishing_returns flag',
]:
    bullet(item)

doc.add_heading('Exp G (Transfer to GPT-4o-mini)', level=2)
doc.add_paragraph(
    'File: src/aapo/inference/openai_client.py (new file). Implement OpenAIClient that follows '
    'InferenceBackend protocol from vllm_client.py. Then add to configs/models_transfer.yaml:'
)
code('models:\n'
     '  target:\n'
     '    name: "gpt-4o-mini"\n'
     '    backend: "openai"\n'
     '    api_key_env: "OPENAI_API_KEY"\n'
     '    temperature: 0.3\n'
     '    max_tokens: 512\n'
     '  judge:\n'
     '    # ... (can keep Qwen-30B for judge, or use OpenAI here too)')

doc.add_heading('Exp J (tau-bench Migration)', level=2)
doc.add_paragraph(
    'Major refactor; consider doing this on a feature branch (git checkout -b feature/tau-bench).'
)
for item in [
    'Replace src/aapo/environment/database.py with tau-bench airline environment loader',
    'Replace src/aapo/environment/tools.py with tau-bench tool definitions',
    'Replace src/aapo/environment/policies.py with tau-bench policy taxonomy',
    'Adapt src/aapo/environment/target_agent.py to use tau-bench target (or call their API)',
    'Update src/aapo/auditor/scenarios.py: ScenarioPool.from_tau_bench()',
    'Update src/aapo/reward/verifiers.py: rewrite verifiers for tau-bench schema',
    'Verify reward components still make sense; some adaptations likely needed for tau-bench reward',
]:
    bullet(item)

doc.add_page_break()

# =============================================================
# 11. DATA SYNC
# =============================================================
doc.add_heading('11. Data Sync and Result Aggregation', level=1)

doc.add_heading('Per-Cluster Output Directory', level=2)
doc.add_paragraph(
    'Each cluster writes to its local data/ directory. Periodically aggregate to a central NFS:'
)
code('# From any cluster, push results to central\n'
     'rsync -av data/logs/   central_nfs:/shared/aapo/results/{cluster_name}/logs/\n'
     'rsync -av data/eval/   central_nfs:/shared/aapo/results/{cluster_name}/eval/\n'
     '\n'
     '# Pull all results to local for plotting\n'
     'rsync -av central_nfs:/shared/aapo/results/  ./results_combined/\n'
     'python paper/generate_figures.py --input-dir ./results_combined')

doc.add_heading('Result Aggregation Script', level=2)
doc.add_paragraph(
    'paper/generate_figures.py reads from data/logs/ and data/eval/ by default. To consume '
    'multi-cluster results, modify the BASE constant or pass --input-dir flag.'
)

doc.add_heading('Naming Conventions', level=2)
table(
    ['Component', 'Pattern', 'Example'],
    [
        ['Training run dir', 'data/logs/{method}_seed{N}_{tag}/', 'data/logs/dpo_seed42_v3/'],
        ['Checkpoint dir', 'data/checkpoints/{run_name}/step_{N}/', 'data/checkpoints/dpo_seed42_v3/step_300/'],
        ['Eval report', 'data/eval/{run_name}_step_{N}_report.json', 'data/eval/dpo_seed42_v3_step_300_report.json'],
        ['Eval episodes', 'data/eval/{run_name}_step_{N}_logs/episodes.jsonl', '...'],
    ]
)

doc.add_page_break()

# =============================================================
# 12. RISK REGISTER
# =============================================================
doc.add_heading('12. Risk Register and Common Failures', level=1)

doc.add_heading('Known Failure Modes', level=2)
table(
    ['Symptom', 'Likely Cause', 'Mitigation'],
    [
        ['vLLM Server not ready after 600s',
         'Bad node, port conflict, or OOM during model load',
         'Resubmit excluding bad nodes (gpu-13, gpu-57)'],
        ['CUDA OOM during training',
         'Sequence length too long at higher k_max',
         'Already mitigated with serial fwd pass + gradient checkpointing'],
        ['Tokenizer "Already borrowed"',
         'Concurrent tokenizer access',
         'Already mitigated with threading.Lock in model.py'],
        ['Generation hangs for hours',
         'use_cache lost after gradient checkpointing',
         'Already mitigated with explicit use_cache=True restoration'],
        ['kl_loss = 0.0000 in GRPO logs',
         'KL term may not be wired correctly',
         'Action: audit grpo.py update() logic'],
        ['Empty training_metrics.jsonl',
         'Job crashed before first log flush',
         'Check .err file for startup error; verify HF cache populated'],
        ['Coverage drops at later checkpoints',
         'Model converges to exploiting easy policies',
         'V3 diminishing returns + novelty schedule address this'],
    ]
)

doc.add_heading('Compute Budget Risk', level=2)
doc.add_paragraph(
    'Each training run = 72h x 2 GPUs = 144 GPU-hours. Each eval = ~6-12h x 2 GPUs = 12-24 GPU-hours. '
    'Full plan (V3 evals + ablation + 3 seeds + transfer + tau-bench) = ~2,500 GPU-hours. '
    'On a single 8-job-limit cluster this is 5-6 weeks of wallclock; with two clusters, ~3 weeks.'
)

doc.add_heading('Submission Risk', level=2)
doc.add_paragraph('NeurIPS 2026 submission deadline (typically late May 2026). Critical items:')
for item in [
    'Decide tau-bench switch within next 7 days (longer = unrecoverable)',
    'Get GRPO KL audit done in week 1 (may invalidate algorithm comparison)',
    'Lock down framing (system vs reward thesis) before writing',
    'Reserve last 7 days for paper writing, figures, NeurIPS checklist',
]:
    bullet(item)

doc.add_page_break()

# =============================================================
# 13. TIMELINE
# =============================================================
doc.add_heading('13. Submission Timeline', level=1)

doc.add_paragraph('Working backward from a hypothetical late-May 2026 deadline:')

table(
    ['Week', 'Goal', 'Cluster Allocation'],
    [
        ['Week -6', 'V3 eval complete; GRPO KL audit; tau-bench decision', 'Long+CSCC'],
        ['Week -5', 'tau-bench env integration if switching; otherwise dedup ablation training', 'Long+External'],
        ['Week -4', 'Full factorial 3 algos x 2 reward versions training', 'Long+CSCC+External'],
        ['Week -3', 'Three-seed runs on main cells; transfer experiment training', 'Long+External'],
        ['Week -2', 'Eval all checkpoints; B0+B2 baselines run (200 ep)', 'CSCC+External'],
        ['Week -1', 'Paper writing: figures, tables, related work, supervisor revisions', '-'],
        ['Submission', 'Final compile, checklist, NeurIPS upload', '-'],
    ]
)

doc.add_heading('What NOT to do', level=2)
for item in [
    'Do not run additional V1 experiments. V1 was a baseline; we have enough data to characterize it.',
    'Do not extend V2 or V3 training beyond their current step count. Both have converged.',
    'Do not chase additional reward components beyond what V3 already has. Time is better spent on ablations.',
    'Do not submit to NeurIPS without 3 seeds on main cells; reviewers will reject single-seed cross-algo.',
    'Do not delay the tau-bench decision past Week -5. Deeper into the timeline = irrecoverable.',
]:
    bullet(item)

# =============================================================
# SAVE
# =============================================================
import os
out = '/home/salem.lahlou/kmishra/aapo/AAPO_Experiment_Plan.docx'
doc.save(out)
print(f'Saved to {out}')
print(f'Size: {os.path.getsize(out)/1024:.1f} KB')
