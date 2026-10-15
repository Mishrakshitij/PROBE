# INTERNAL — camera-ready issues. NOT for any rebuttal post.

Everything here was found by us, not raised by a reviewer. Posting any of it during
discussion would open new attack surface on points nobody is currently contesting.
Fix these quietly in the camera-ready.

---

## 1. Same-base ±training control on SkyAssist / Gemma-4 — do not post

We ran the untrained Qwen3-4B base through the identical pipeline (verified exact LoRA
identity, 288 tensors) as a ±training control. It does not support the framing in the
posted rebuttal and reproduces exactly Xtwt's W2 complaint.

| Target | Scoring | zero-shot 4B | PROBE | Δ |
|---|---|---|---|---|
| SkyAssist | deterministic | 50% (10/20), 174 v/100 | GRPO 55%, 216 | +1 policy |
| SkyAssist | judge-included | 55% (11/20), 1028 v/100 | GRPO 60%, 1176 | +1 policy (S2) |
| Gemma-4 | deterministic | 35% (7/20), 176 v/100 | DPO 40%, 190 | +1 policy (B4) |
| Gemma-4 | judge-included | 45% (9/20), 426 v/100 | DPO 45%, 436 | **tie, identical policy set** |

Bootstrap (deterministic, 50 ep/arm): GRPO − zero-shot = +5 pp, CI [−5, +15], p = 0.38;
yield/ep +0.42, CI [−0.72, +1.60], p = 0.24. DPO − zero-shot = +5 pp, CI [−10, +15],
p = 0.53. **Not statistically distinguishable at 50 episodes.**

Caveats that matter for the camera-ready framing: this control inherits PROBE's
scenario pool and auditing system prompt, so it isolates the RL step only, not the
system. Direction is consistent (+1 policy on both targets, higher entropy, lower top-2
concentration on both) but underpowered — separating +5 pp would need ~200 episodes per
arm.

Reports: `data/eval/zeroshot_4b_base_report.json`,
`data/eval/zeroshot_4b_transfer_gemma4_report.json`.

The same-base control that *is* postable, and is already posted, is τ-bench: zero-shot
Qwen3.5-9B 20% / 13 violations vs PROBE-DPO on the identical base 30% / 29 violations.

## 2. `main.tex:213` mislabels the zero-shot baseline

Table row says "B2 (zero-shot Qwen3-4B)"; `main.tex:195` says Qwen3-30B.
`scripts/run_baseline.py:122` loads the auditor from the **target** role — it is the
**30B**. Fix the table label.

## 3. Abstract's "higher-yield" claim does not survive judge-free scoring

Abstract claims PROBE finds "broader **and higher-yield**" violations than zero-shot.
Deterministically the *breadth* claim holds (11/20 vs 9/20; entropy 3.23 vs 1.90; top-2
concentration 29% vs 83%) but *raw yield* does not (216 vs 591 — the zero-shot 30B
re-finds the same two policies, B2 and S4, in nearly every episode; 82% of PROBE's
pre-removal violations were judge hits). Revise to a dedup-aware / distinct-policy
framing, which is what the dedup reward was designed around, rather than raw count.

## 4. The V1→V2→V3 narrative needs revising under judge-free scoring

At matched optimiser, deterministic: GRPO 50% (V2) vs 50–55% (V3); DPO 55% vs 55%;
GFlowNet V3 35% is *worse* than V2 40%. The V2→V3 gains in the submission were
concentrated in the hybrid judge policies — consistent with 82% of PROBE's pre-removal
violations being judge hits. (Judge-included, the ordering does hold: GRPO 55→60,
DPO 55→60 — which is why the round-1 rebuttal's numbers are internally consistent.)
The claim that survives judge-free is narrower and still ours: policy-specific coverage
credit beats generic diversity (50–55% vs 35–45%). Rewrite Section 5 and the V3
discussion to that claim rather than to a monotone V1→V2→V3 improvement.

## 5. Eval-log hygiene

Several `data/eval/*_logs/episodes.jsonl` concatenate multiple eval runs across dates,
some including pre-2026-05-02 episodes from the invalidated LoRA-loading era (e.g.
`dpo_seed42_v2_step_50` = 44 + 56 + 50 episodes across three dates). Union coverage
grows with episode count, so re-scoring a whole file inflates it. All numbers in
`rebuttal_followup.md` use single-date, 50-episode, post-fix groups. Re-scoring scripts
released with the paper must filter explicitly rather than glob whole files.
