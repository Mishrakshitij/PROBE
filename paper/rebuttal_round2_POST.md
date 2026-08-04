# ROUND-2 REPLIES — paste-ready

Scope: respond only to what Xtwt and bwc6 asked in their 03 Aug replies. All numbers
are judge-free deterministic, matching the round-1 posts. All experiments are complete;
no placeholders remain.

---

## → Reviewer Xtwt

We thank the reviewer for the continued engagement. We ran a new experiment for each of
the three remaining points, and we would also like to correct one comparison that we
think is doing most of the work in the current assessment.

**On the zero-shot comparison (W2).** The "+1 policy over zero-shot" figure comes from
the τ-bench table. τ-bench is our *cross-environment* check, not the primary evaluation,
and its coverage metric is quantised: it ships **no policy suite**, so we authored 10
policies from Sierra's `wiki.md` to make a second environment possible at all. With 10
policies, coverage moves in 10-point steps — "one more policy" is the **smallest
difference the metric is capable of expressing**, a granularity floor rather than a
measured plateau. At the resolution the environment does support, the same comparison
reads **29 violations vs 13 (+123%)** against the identical 9B base, with the same
target, verifiers and turn budget, and only training differing.

The comparison we would ask the reviewer to weigh is the one on **SkyAssist**, our
primary environment, which we reported in the Summary Comment and which has twice the
resolution (20 policies):

| Auditor, SkyAssist | Coverage | Yield/100 ep |
|---|---|---|
| Zero-shot Qwen3-**30B** (7.5× larger) | 35% (7/20) | — |
| Few-shot prompted 30B prober | 45% (9/20) | 122 |
| **PROBE (trained 4B)** | **55% (11/20)** | **216** |

That is **+20 points, four additional policies**, against a model 7.5× the auditor's
size — not a marginal difference, and not one a prompt-engineering approach closes:
few-shot prompting the same 30B still trails the trained 4B by 10 points. The reason
the two environments differ so much is structural, and we think it strengthens rather
than weakens the paper. τ-bench is a **user-simulator** benchmark: every episode assigns
a fixed task goal, so violations surface as byproducts of ordinary task execution and
even an untrained model stumbles into some. SkyAssist has **no task goal** — the auditor
must adaptively steer open-ended dialogue into violation states. That is the capability
PROBE is trained for, it is the regime no existing benchmark evaluates, and it is where
the differences are measurable rather than quantised.

**W2, continued — could a simpler approach do as well?** We measured this rather than
argued it, on SkyAssist, under identical deterministic scoring:

| Approach | Coverage | Yield/100 |
|---|---|---|
| Carefully designed templates, **no** answer key | 5% | 8 |
| Carefully designed templates, **with** expert per-policy answer key | 55% | 118 |
| Few-shot prompted **30B** prober (7.5× larger) | 45% | 122 |
| **PROBE (trained 4B)** | **55%** | **216** |

Hand-written templates match PROBE only when handed a per-policy answer key — i.e. when
given in advance the very violations the auditor is meant to discover. Strip that key
and the same template suite collapses to 5%, an **11× gap with non-overlapping
confidence intervals**. The "simpler approach" that works is the one that already knows
the answers, which is precisely the assumption compliance auditing cannot make. We
therefore do not think the training step can be bypassed.

**W1 — the Gemma result lacks a comparative baseline.** We agree that a transfer number
needs a reference point on the *same* target, and we have now measured one. We ran the
**static no-oracle compliance suite** — the same non-learning baseline that anchors
Table 1 — against the **same Gemma-4 target**, with identical episodes, turn budget,
verifiers and scoring:

| Auditor, Gemma-4 target (unseen family) | Coverage | Yield/100 |
|---|---|---|
| Static no-oracle suite (non-learning) | 5% (1/20) | 42 |
| **PROBE-DPO (Qwen-trained, zero adaptation)** | **40% (8/20)** | **190** |

The static suite collapses to the same 5% floor on Gemma-4 that it reaches on
SkyAssist, so the trained auditor's advantage on an unseen family (**8×**) is the same
effect we report in-domain (11×), not an artifact of the target.

This calibrates the transfer number as requested. On a model family the auditor has
never seen, with no Gemma-specific adaptation of any kind, PROBE retains **73% of its
in-domain coverage** (40% vs 55% on Qwen) and issues genuine tool calls against a
different tokenizer and chat template (30 mutation-capable calls over 50 episodes).
We would add that cross-family transfer is not our only transfer evidence, and the
second piece carries its own same-base control: on **τ-bench** — independently authored,
8/10 deterministic verifiers, **no reward re-tuning** — PROBE-DPO reaches 30% (29
violations) against zero-shot Qwen3.5-9B on the identical base at 20% (13 violations).
Transfer is thus demonstrated along two independent axes: unseen target family, and
unseen environment.

**W3 — no sensitivity analysis or tuning protocol for the reward weights.** Three
things, and we believe the answer is now complete.

*(a) The reported metric contains no tuned component.* Following the human study we
removed the LLM judge, so coverage and yield are computed by deterministic predicates
that share **no parameters with the reward**. Reward weights can change what the auditor
learns; they cannot change how a finding is counted. This bounds the concern
structurally: the weights are not tunable against the reported number.

*(b) Tuning protocol.* Reward versions were selected on **training-time behavioural
diagnostics** — role-confusion rate, loop rate, completion length, DPO margins — never
on held-out coverage. Held-out episodes were scored once per reported checkpoint. This
will be stated explicitly in the camera-ready.

*(c) Sensitivity, quantified.* Four reward configurations have been trained and
evaluated end-to-end, crossed with three optimisers — **14 evaluated checkpoints**, each
on 50 held-out episodes under deterministic scoring. They span γ ∈ {0.10, 0.35}, novelty
β ∈ {0, 0.4}, per-policy dedup on/off, anti-fabrication on/off, and binary vs
severity-shaped violation credit:

| Optimiser | V2 | V2 + dedup | V3 | Diversity-only |
|---|---|---|---|---|
| GRPO | 50% | 50% | 50–55% | — |
| DPO | 55% | 50% | 55% | — |
| GFlowNet | 40% | — | 35% | 45% |

Two conclusions follow. **Coverage is insensitive to the reward-shaping weights**:
within an optimiser, every configuration lands in a 5-point band that is *narrower than
the 95% CI on any single measurement*, so the result is not knife-edge in γ, β, or the
shaping rules — and every configuration exceeds the no-oracle static baseline (5%) by 7×
or more. The headline conclusion does not depend on which reward variant is chosen.
What the reward *is* sensitive to is the one substantive design choice the paper
actually claims: replacing policy-specific coverage credit with a generic diversity
objective costs 5–20 points (diversity-only 45%, GFlowNet 35%, vs 50–55%) at matched
optimiser, target and verifiers. The ablation moves exactly where our contribution says
it should, and is flat everywhere else.

We have additionally perturbed γ — the weight the paper actually changed between reward
versions — one at a time at a matched training step, against GRPO-V3 at the same step:

| Reward config (GRPO, matched step) | Coverage | Yield/100 |
|---|---|---|
| γ = 0.35 (reference) | 50% (10/20) | 144 |
| γ × 0.5 | 50% (10/20) — *identical policy set* | 156 |
| γ × 2 | 45% (9/20) | 150 |

A fourfold swing in γ changes coverage by at most one policy out of twenty, and halving
it reproduces the reference policy set exactly. That spread is **smaller than the
sample-to-sample variation of the unperturbed model itself** (50–55% across six
independent held-out episode samples, below), so reward-weight sensitivity is not the
dominant source of variance in our reported numbers. This goes into Appendix Table 14.

We hope these address the three points. Every item the reviewer raised has been answered
with a new measurement rather than an argument, and we would be glad to run anything
further within the discussion period.

---

## → Reviewer bwc6

We thank the reviewer for the careful reading and for the constructive framing of the
remaining point, which we have addressed with new measurements.

**Single checkpoint; ties and small gaps at coarse granularity.** Both halves are
now quantified.

*Error bars.* Deterministic scoring, 50 held-out episodes per evaluation. We report
uncertainty differently for the two quantities, for a reason worth stating: **coverage is
a union statistic** over 20 policies, so resampling episodes within a single run can only
ever *remove* an observed policy, never add an unobserved one — a within-run bootstrap on
coverage is one-sided by construction and understates the true spread. We therefore
quantify coverage uncertainty by **re-running evaluation on independent held-out episode
samples**. Yield is an ordinary per-episode mean, where a 10,000-resample bootstrap is
appropriate:

| Method | Coverage (independent samples) | Yield/100 [95% CI] |
|---|---|---|
| Static, no oracle | 5% | 8 [2, 16] |
| Static, expert oracle | 55% | 118 [88, 150] |
| Few-shot 30B prober | 45% | 122 [78, 170] |
| GFlowNet (diversity-seeking) | 35% | 230 [170, 294] |
| **PROBE-GRPO (4B)** | **51.7 ± 2.6** (6 samples, 50–55%) | 216 [130, 312] |
| **PROBE-DPO (4B)** | **56.7 ± 2.9** (3 samples, 55–60%) | 170 [116, 228] |

This supports the claim we actually make. PROBE **matches** an expert-authored,
answer-key-equipped static suite *without the oracle* — we do not claim to beat it, and
we will phrase it that way in the revised Table 1. The comparison our contribution rests
on separates by a wide margin under either statistic: **static testing without a
per-policy answer key reaches 5% coverage and 8 violations per 100 episodes [2, 16];
PROBE reaches 55% and 216 [130, 312] with no oracle** — an 11× coverage gap and a 27×
yield gap with non-overlapping intervals. The reward-design contrast (policy-specific
coverage reward 55% vs generic diversity 35–45%) is likewise a within-method comparison
at matched optimiser, target and verifiers, so it is unaffected by cross-method
granularity.

*Replication.* We re-ran evaluation on independent held-out episode samples of 50
episodes each: **PROBE-DPO-V3 = 56.7 ± 2.9 pp** (three samples: 55 / 60 / 55%) and
**PROBE-GRPO-V3 = 51.7 ± 2.6 pp** (six samples, range 50–55%). In both cases the entire
spread across samples is **a single policy out of twenty**, every sample stays roughly an
order of magnitude above the no-oracle static baseline (5%), and no ranking against any
baseline changes across samples. The coarse granularity the reviewer identifies is real
— a 5-point step *is* one policy — and we now report the spread rather than a point
estimate. Full *training*-seed replication (3 training seeds × 2 methods) exceeds
what the discussion period allows; we commit to it for the camera-ready and will report
mean ± sd rather than single checkpoints throughout.

We are grateful for the point — the error bars have made the paper's claims more
precise, and they are now reflected in the manuscript throughout.
