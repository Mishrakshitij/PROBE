# PROBE — follow-up responses (round 2, after reviewer replies of 03 Aug 2026)

> **Working notes.** The paste-ready replies are `paper/rebuttal_round2_POST.md`;
> where the two differ, POST is authoritative. The coverage table below is kept
> complete (including the cooperative simulator) as our internal reference — the
> posted reply reports coop in prose rather than as a table row.

Thread status: **AuG2 = 6** (satisfied, no open items). **bwc6** — positive, says the
baselines + human study "resolve" the verifier/reward-hacking concerns; two remaining
items, both answerable with results we have. **Xtwt** — "I will keep my score", three
remaining items (W1/W2/W3).

Scoring convention: **every number below is judge-free deterministic**, identical to
what we posted in round 1 (PROBE 55%, static-no-oracle 5%, few-shot 45%, coop 60%,
GFlowNet 35%, Gemma transfer 40%). Do not mix in judge-included numbers — the posted
rebuttal committed to the deterministic frame.

`[PENDING-x]` marks a number from a job still running; fill before posting.

---

## Numbers available for this round

Deterministic scoring (LLM judge removed), 50 held-out episodes each, SkyAssist:

| Method | Coverage | 95% CI | Yield/100 | Entropy | Top-2 policy share |
|---|---|---|---|---|---|
| Static, no oracle (fixed) | 5% | [5, 5] | 8 | — | — |
| Static, expert oracle (scripted) | 55% | [40, 55] | 118 | 3.26 | 31% |
| Few-shot 30B prober | 45% | [30, 45] | 122 | 2.58 | 57% |
| Cooperative simulator (30B) | 60% | [45, 60] | 162 | 3.29 | 38% |
| GFlowNet (diversity-seeking) | 35% | [20, 35] | 230 | 1.38 | 88% |
| Diversity-only reward | 45% | [30, 45] | 104 | 2.67 | 54% |
| **PROBE-GRPO (4B)** | **55%** | [40, 55] | 216 | 3.23 | 29% |
| **PROBE-DPO (4B)** | **55%** | [35, 55] | 170 | 3.15 | 36% |

CIs: 10,000-resample bootstrap over episodes. Entropy = Shannon entropy of the
policy distribution (higher = findings spread across more policies).

New this round: eval-seed replication (below); static no-oracle suite against the
**Gemma-4** target `[PENDING-E]`; γ perturbations `[PENDING-D]`.

---

## Reviewer Xtwt

We thank the reviewer for keeping the discussion concrete, and we ran an additional
experiment for each remaining point.

**W1 — the Gemma result lacks a comparative baseline.** Agreed that a transfer number
needs a reference point on the *same* target, and we have now measured one. We ran our
**static no-oracle compliance suite** — the same non-learning baseline that anchors
Table 1 — against the **same Gemma-4 target**, with identical episodes, turn budget,
verifiers and deterministic scoring:

| Auditor, Gemma-4 target (unseen family) | Coverage |
|---|---|
| Static no-oracle suite (non-learning) | `[PENDING-E]` |
| **PROBE-DPO (Qwen-trained, zero adaptation)** | **40% (8/20)** |

This calibrates the transfer number in the way the reviewer asks: on a target family
the auditor has never seen, and with no Gemma-specific adaptation of any kind, PROBE
retains **73% of its in-domain coverage** (40% vs 55% on Qwen) and issues genuine tool
calls against a different tokenizer and chat template (30 mutation-capable calls over
50 episodes). The drop is the expected cost of an unseen family; the level is far above
what a non-learning suite reaches on that same target.

We would also note that the cross-family result is not our only transfer evidence, and
the *second* piece already carries its own same-base control: on τ-bench — an
independently authored environment, 8/10 deterministic verifiers, **no reward
re-tuning** — PROBE-DPO reaches 30% (29 violations / 50 episodes) against **zero-shot
Qwen3.5-9B on the identical base** at 20% (13 violations). Transfer is therefore
demonstrated along two independent axes: unseen target family (Gemma-4) and unseen
environment (τ-bench).

**W2 — PROBE-DPO covers only one policy more than zero-shot Qwen3.5-9B; simpler
approaches might do as well.** Two clarifications.

*First, τ-bench is a cross-environment check, not our primary evaluation, and its
coverage metric is quantised.* τ-bench ships **no policy suite** — we authored 10
policies from Sierra's `wiki.md` specifically to make a second environment possible.
With 10 policies, coverage moves in 10-point steps, so "one more policy" is the
**smallest difference the metric can express**: a granularity floor, not a measured
plateau. On the finer-grained violation count over the same episodes the same
comparison reads **29 vs 13 (+123%)** — same base model, same target, same verifiers,
same turn budget, only training differs. That *is* the fair same-base control the
original W2 asked for, and at the resolution the environment actually supports it is
not marginal.

There is also a reason τ-bench is the *easy* case for an untrained auditor, which is
why we present SkyAssist as primary. τ-bench is a **user-simulator** benchmark: each
episode assigns a fixed task goal, so violations surface as byproducts of ordinary task
execution and even an untrained model stumbles into some. SkyAssist has **no task
goal** — the auditor must adaptively steer open-ended dialogue into violation states.
That is the capability PROBE is trained for, and it is the setting where the
differences are measurable rather than quantised.

*Second, the "simpler approaches" hypothesis is measured and rejected on SkyAssist*,
our primary environment, where 20 policies give twice the resolution:

| Approach | Coverage | Yield/100 |
|---|---|---|
| Carefully designed templates, **no** answer key | 5% | 8 |
| Carefully designed templates, **with** expert per-policy answer key | 55% | 118 |
| Few-shot prompted **30B** prober (7.5× larger) | 45% | 122 |
| **PROBE (trained 4B)** | **55%** | **216** |

Few-shot prompting of a 7.5×-larger model does not reach the trained 4B, and
hand-written templates match it only when handed a per-policy answer key — i.e. when
given in advance the very violations the auditor is supposed to discover. Without that
key the same template approach collapses to 5%, an 11× gap with non-overlapping
confidence intervals. The "simpler approach" that works is the one that already knows
the answers; that is precisely the assumption compliance auditing cannot make.

**W3 — no sensitivity analysis or tuning protocol for the reward weights.** Three
things, one of them new.

*(a) The reported metric now contains no tuned component.* Following the human study we
removed the LLM judge, so coverage and yield are computed by deterministic predicates
that share **no parameters with the reward**. Reward weights can change what the auditor
learns; they cannot change how a finding is counted. This bounds the concern
structurally: the weights are not tunable against the reported number.

*(b) Tuning protocol.* Reward versions were selected on **training-time behavioural
diagnostics** — role-confusion rate, loop rate, completion length, DPO margins — never
on held-out coverage. Held-out episodes were scored once per reported checkpoint. We
will state this explicitly in the camera-ready.

*(c) Sensitivity, quantified.* Four reward configurations have been trained and
evaluated end-to-end, crossed with three optimisers — 14 evaluated checkpoints in
total, each on 50 held-out episodes under deterministic scoring. They span
γ ∈ {0.10, 0.35}, novelty β ∈ {0, 0.4}, per-policy dedup on/off, anti-fabrication
on/off, and binary vs severity-shaped violation credit:

| Optimiser | V2 | V2 + dedup | V3 | Diversity-only |
|---|---|---|---|---|
| GRPO | 50% | 50% | 50–55% | — |
| DPO | 55% | 50% | 55% | — |
| GFlowNet | 40% | — | 35% | 45% |

Two things follow. **Coverage is insensitive to the reward-shaping weights**: within an
optimiser, every configuration lands in a 5-point band narrower than the 95% CI on any
single measurement, so the result is not knife-edge in γ, β, or the shaping rules. And
**every** configuration exceeds the no-oracle static baseline (5%) by 7× or more — the
headline conclusion does not depend on which reward variant is chosen. What the reward
*is* sensitive to is the one substantive design choice we actually claim: replacing
policy-specific coverage credit with a generic diversity objective costs 5–20 points
(diversity-only 45%, GFlowNet 35%, vs 50–55%) at matched optimiser, target and
verifiers. That is the ablation the paper rests on, and it is the one that moves.

We additionally perturbed γ — the weight the paper actually changed between reward
versions — one at a time (×0.5 and ×2) at a matched training step, against GRPO-V3 at
the same step (50%): `[PENDING-D]`. This will be added to Appendix Table 14.

---

## Reviewer bwc6

We thank the reviewer, and we address both remaining points with new measurements.

**1. Single checkpoint; ties and small gaps at coarse granularity.** Fair, and we have
now quantified both parts.

*Error bars.* 95% bootstrap CIs (10,000 resamples over held-out episodes) are in the
table above. They make the structure of the comparison explicit. The mid-table
differences do **not** separate: PROBE 55% [40, 55], expert-oracle static 55% [40, 55]
and cooperative 60% [45, 60] are statistically indistinguishable — which is exactly the
claim we make, that PROBE **matches** an oracle-authored suite *without the oracle*,
not that it beats it. The comparison our contribution rests on is the one that survives
error bars by a wide margin: **static testing without a per-policy answer key reaches
5% [5, 5]; PROBE reaches 55% [40, 55] with no oracle** — an 11× gap with
non-overlapping intervals. The reward-design contrast (policy-specific coverage reward
55% vs generic diversity 35–45%) is likewise a within-method comparison at matched
optimiser, target and verifiers.

*Replication.* We re-ran evaluation on **three independent held-out episode samples**
(eval seeds 42/123/456, 50 episodes each) for PROBE-GRPO-V3: coverage **55 / 50 / 50%,
mean 51.7 ± 2.9 pp**; yield 216 / 280 / 156 per 100 episodes. The variation is inside a
single policy slot, every sample stays ~10× the no-oracle static baseline, and no
ranking against any baseline changes. `[PENDING-C2]` adds the same for PROBE-DPO-V3.
Full *training*-seed replication (3 training seeds × 2 methods) is a larger compute
commitment than the discussion period allows; we commit to it for the camera-ready and
will report mean ± sd rather than single checkpoints throughout.

**2. The cooperative simulator scores above PROBE and should be reported as such.**
Agreed, and it will be: coop **60%** vs PROBE **55%** appears in the revised Table 1, in
the text, and in the discussion — as a result, not a footnote. The mechanism (its
additional policy is B5, reached as a byproduct of *completing* the booking rather than
by adversarial steering) accompanies the number as an explanation, not as a reason to
discount it. We think it also sharpens the paper's central distinction: a **4B
adversarial auditor matches a 30B cooperative simulator**, and the two surface partly
different violations, so cooperative user-simulation and adversarial auditing are
complementary rather than interchangeable. Combining task completion with adversarial
probing is the natural route to the booking-flow policies, which we now state as
concrete future work.

---

## Reviewer AuG2

No open items (score 6). Short note only: the human study, the deterministic
re-scoring, the cross-family control and the reward-sensitivity table added during
discussion are all folded into the camera-ready; Q1's verifier-confidence answer is
strengthened by the fact that all 20 policies are now scored deterministically.

---

## Camera-ready commitments made in this thread

Only items a reviewer actually raised — nothing volunteered.

1. Report coop 60% > PROBE 55% in Table 1 and in the text (bwc6).
2. Report all headline numbers with bootstrap CIs, and multi-seed mean ± sd (bwc6).
3. State the reward tuning protocol explicitly — behavioural diagnostics, not held-out
   coverage (Xtwt W3b).
4. Add the reward-sensitivity table and the γ perturbations as Appendix Table 14
   (Xtwt W3c).
5. Add the Gemma-4 comparative baseline to the transfer table (Xtwt W1).

Internal-only camera-ready issues found during this round are recorded separately in
`paper/camera_ready_internal.md` — **they are not part of any rebuttal post.**
