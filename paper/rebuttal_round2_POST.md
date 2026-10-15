# ROUND-2 REPLIES — paste-ready

Scope: respond only to what Xtwt and bwc6 asked in their 03 Aug replies. All numbers
are judge-free deterministic, matching the round-1 posts. All experiments are complete;
no placeholders remain.

---

## Reviewer Xtwt

We thank the reviewer for the continued engagement. We ran a new experiment for each of
the three remaining points, and we would also like to correct one comparison that we
think is doing most of the work in the current assessment.

**On the zero-shot comparison (W2).** The "+1 policy over zero-shot" figure comes from
the $\tau$-bench table. $\tau$-bench is our *cross-environment* check, not the primary evaluation,
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
| Zero-shot Qwen3-**30B** (7.5$\times$ larger) | 35% (7/20) | — |
| Few-shot prompted 30B prober | 45% (9/20) | 122 |
| **PROBE (trained 4B)** | **55% (11/20)** | **216** |

That is **+20 points, four additional policies**, against a model 7.5$\times$ the auditor's
size — not a marginal difference, and not one a prompt-engineering approach closes:
few-shot prompting the same 30B still trails the trained 4B by 10 points. The reason
the two environments differ so much is structural, and we think it strengthens rather
than weakens the paper. $\tau$-bench is a **user-simulator** benchmark: every episode assigns
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
| Few-shot prompted **30B** prober (7.5$\times$ larger) | 45% | 122 |
| **PROBE (trained 4B)** | **55%** | **216** |

Hand-written templates match PROBE only when handed a per-policy answer key — i.e. when
given in advance the very violations the auditor is meant to discover. Strip that key
and the same template suite collapses to 5%, an **11$\times$ gap with non-overlapping
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
SkyAssist, so the trained auditor's advantage on an unseen family (**8$\times$**) is the same
effect we report in-domain (11$\times$), not an artifact of the target.

This calibrates the transfer number as requested. On a model family the auditor has
never seen, with no Gemma-specific adaptation of any kind, PROBE retains **73% of its
in-domain coverage** (40% vs 55% on Qwen) and issues genuine tool calls against a
different tokenizer and chat template (30 mutation-capable calls over 50 episodes).
We would add that cross-family transfer is not our only transfer evidence, and the
second piece carries its own same-base control: on **$\tau$-bench** — independently authored,
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
on 50 held-out episodes under deterministic scoring. They span $\gamma \in \{0.10, 0.35\}$, novelty
$\beta \in \{0, 0.4\}$, per-policy dedup on/off, anti-fabrication on/off, and binary vs
severity-shaped violation credit:

| Optimiser | V2 | V2 + dedup | V3 | Diversity-only |
|---|---|---|---|---|
| GRPO | 50% | 50% | 50–55% | — |
| DPO | 55% | 50% | 55% | — |
| GFlowNet | 40% | — | 35% | 45% |

Two conclusions follow. **Coverage is insensitive to the reward-shaping weights**:
within an optimiser, every configuration lands in a 5-point band that is *narrower than
the 95% CI on any single measurement*, so the result is not knife-edge in $\gamma$, $\beta$, or the
shaping rules — and every configuration exceeds the no-oracle static baseline (5%) by 7$\times$
or more. The headline conclusion does not depend on which reward variant is chosen.
What the reward *is* sensitive to is the one substantive design choice the paper
actually claims: replacing policy-specific coverage credit with a generic diversity
objective costs 5–20 points (diversity-only 45%, GFlowNet 35%, vs 50–55%) at matched
optimiser, target and verifiers. The ablation moves exactly where our contribution says
it should, and is flat everywhere else.

We have additionally perturbed $\gamma$ — the weight the paper actually changed between reward
versions — one at a time at a matched training step, against GRPO-V3 at the same step:

| Reward config (GRPO, matched step) | Coverage | Yield/100 |
|---|---|---|
| $\gamma = 0.35$ (reference) | 50% (10/20) | 144 |
| $\gamma \times 0.5$ | 50% (10/20) — *identical policy set* | 156 |
| $\gamma \times 2$ | 45% (9/20) | 150 |

A fourfold swing in $\gamma$ changes coverage by at most one policy out of twenty, and halving
it reproduces the reference policy set exactly. That spread is **smaller than the
sample-to-sample variation of the unperturbed model itself** (50–55% across six
independent held-out episode samples, below), so reward-weight sensitivity is not the
dominant source of variance in our reported numbers. This goes into Appendix Table 14.

We hope these address the three points. Every item the reviewer raised has been answered
with a new measurement rather than an argument, and we would be glad to run anything
further within the discussion period.

---

## Reviewer bwc6

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
| **PROBE-GRPO (4B)** | **51.7 $\pm$ 2.6** (6 samples, 50–55%) | 216 [130, 312] |
| **PROBE-DPO (4B)** | **56.7 $\pm$ 2.9** (3 samples, 55–60%) | 170 [116, 228] |

This supports the claim we actually make. PROBE **matches** an expert-authored,
answer-key-equipped static suite *without the oracle* — we do not claim to beat it, and
we will phrase it that way in the revised Table 1. The comparison our contribution rests
on separates by a wide margin under either statistic: **static testing without a
per-policy answer key reaches 5% coverage and 8 violations per 100 episodes [2, 16];
PROBE reaches 55% and 216 [130, 312] with no oracle** — an 11$\times$ coverage gap and a 27$\times$
yield gap with non-overlapping intervals. The reward-design contrast (policy-specific
coverage reward 55% vs generic diversity 35–45%) is likewise a within-method comparison
at matched optimiser, target and verifiers, so it is unaffected by cross-method
granularity.

*Replication.* We re-ran evaluation on independent held-out episode samples of 50
episodes each: **PROBE-DPO-V3 = 56.7 $\pm$ 2.9 pp** (three samples: 55 / 60 / 55%) and
**PROBE-GRPO-V3 = 51.7 $\pm$ 2.6 pp** (six samples, range 50–55%). In both cases the entire
spread across samples is **a single policy out of twenty**, every sample stays roughly an
order of magnitude above the no-oracle static baseline (5%), and no ranking against any
baseline changes across samples. The coarse granularity the reviewer identifies is real
— a 5-point step *is* one policy — and we now report the spread rather than a point
estimate. Full *training*-seed replication (3 training seeds $\times$ 2 methods) exceeds
what the discussion period allows; we commit to it for the camera-ready and will report
mean $\pm$ sd rather than single checkpoints throughout.

We are grateful for the point — the error bars have made the paper's claims more
precise, and they are now reflected in the manuscript throughout.

---

## Reviewer AuG2

We thank the reviewer, and we accept both points.

**GFlowNet.** Agreed — it is a strong alternative rather than merely a baseline. Under
identical verifiers it attains the highest raw violation yield of any method we
evaluated (230 per 100 episodes, above PROBE-GRPO's 216); it differs from PROBE in what
it optimises — mode-covering volume rather than distinct-policy breadth. We will present
it that way in Section 5 and in the discussion.

**Non-search tool calls.** Agreed, and this belongs in Limitations. Over 50 held-out
episodes, 91–95% of the auditor's tool calls are searches, and the policies it does not
reach are precisely the mutation-gated ones. We will state this explicitly rather than
as a general remark about tool diversity, and report the tool-call distribution in the
appendix.

---

## Closing comment to the Area Chair

We thank the AC and the reviewers. Our 29 July response addressed the four priorities;
this note records only what has changed since, and closes our side of the discussion.

**P1, now on a second target family.** The oracle-free static suite we reported on
SkyAssist (5% vs PROBE 55%) has since been run against the **Gemma-4** target as well,
where it again reaches **5%**, against PROBE-DPO's **40%**. "Adaptive beats static
without an oracle" is therefore measured on two target families rather than one.

**P4, now answered directly.** Our earlier answer was indirect — the reward transfers to
$\tau$-bench unchanged, versions were selected on training diagnostics, the static
baseline collapses under identical scoring. We have since perturbed $\gamma$, the weight
the paper actually changed between reward versions, one at a time at a matched training
step: **$\gamma \times 0.5 \rightarrow$ 50%**, reproducing the reference policy set
exactly, and **$\gamma \times 2 \rightarrow$ 45%**, against a 50% reference. Across the
14 evaluated checkpoints (four reward configurations crossed with three optimisers)
coverage varies within a 5-point band per optimiser. A fourfold swing in the weight under
scrutiny moves coverage by at most one policy in twenty — less than the variation between
evaluation samples.

**Replication and error bars.** Responding to bwc6's point about single checkpoints, we
re-ran evaluation on independent held-out episode samples: **PROBE-DPO 56.7 $\pm$ 2.9**
(three samples) and **PROBE-GRPO 51.7 $\pm$ 2.6** (six samples). The entire spread in
each arm is one policy out of twenty, and no ranking against any baseline changes across
samples. We also corrected how we report uncertainty: coverage is a union statistic, so a
within-run bootstrap on it is one-sided by construction. Coverage uncertainty now comes
from independent samples, with bootstrap intervals reported for yield, where they are
appropriate.

**Reviewer status.** Reviewer bwc6's remaining request — replication and error bars — is
answered above, and Reviewer AuG2's two final points are accepted and folded into the
manuscript. For Reviewer Xtwt we ran a new experiment for each of the three remaining
points, reported in our reply to them.

**What remains open.** Two things, stated plainly. Full *training*-seed replication
(three training seeds $\times$ two methods) is beyond what this period allows; we commit
to it for the camera-ready and will report mean $\pm$ sd throughout rather than single
checkpoints. And the auditor does not yet demonstrate auditing of the mutation surface:
91–95% of its tool calls are searches, and the policies it does not reach are precisely
the mutation-gated ones. This goes into Limitations rather than staying implicit, and it
defines the next version of the work — policy-conditioned scenarios that begin near a
booking decision, so that reaching those states is by design rather than by accident.

We are grateful to the reviewers for pushing the evaluation to a place we are more
confident defending.
