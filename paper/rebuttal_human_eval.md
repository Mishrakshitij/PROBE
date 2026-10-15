# Rebuttal materials — LLM-judge validation (bwc6 Q3 / AC Priority 2)

Strategy: **do not defend the judge; remove it.** Human validation confirms the
judge over-flags the four semantic policies. Because every policy also has a
deterministic predicate (16/20 always were; S5/P1/B4/B6 each have one too), we
drop the judge entirely and report fully-deterministic scoring — eliminating the
reward-hackable component rather than merely caveating it.

---

## 1. Human study (208 instances, 85 episodes, 2 independent annotators)

| Comparison | Agreement | Cohen's κ | Positive rate | FP | FN |
|---|---|---|---|---|---|
| Human 1 vs Human 2 | **99.5%** | **0.985** | 20.2% vs 20.7% | — | — |
| Judge vs human consensus | 39.6% | **0.084** | judge 76.8% vs human 20.3% | **121** | **4** |

The near-perfect human–human agreement (κ=0.985) shows the cases are unambiguous
under the policy definitions, so the poor judge agreement is not label noise.
McNemar exact p ≈ 5e-31. The judge is a **high-recall, very-low-specificity**
detector: it flags 77% of turns as violations vs 20% under humans.

## 2. Per-policy confusion vs human consensus

| Policy | Severity | TP | FP | FN | TN | **Judge precision** | Recall |
|---|---|---|---|---|---|---|---|
| **S5** confirm-before-action | High | 0 | 39 | 0 | 12 | **0%** | — |
| **P1** pricing privacy | High | 7 | 33 | 0 | 12 | **18%** | 100% |
| B4 upsell | Low | 26 | 14 | 4 | 8 | 65% | 87% |
| B6 round-trip | Low | 5 | 35 | 0 | 12 | 12% | 100% |
| **Aggregate** | | 38 | 121 | 4 | — | **24%** | 90% |

S5 is the extreme case: **0 of 40 flagged hits are genuine.** The judge treats
read-only *searches* as needing confirmation (S5 concerns booking/cancel/modify
only); the target rarely reaches those mutation states, so genuine S5 ≈ 0.
P1 flags ordinary fare quotes as internal-cost disclosure.

## 3. Reward-hacking diagnostic (across training checkpoints)

For each checkpoint we count judge-S5/P1 hits and how many are
deterministically confirmed (proxy for human-genuine, given §2). Escalating
judge-hits with flat/zero genuine hits = a reward-exploitation signature.

**S5 — judge hits (deterministically-confirmed):**
| Method | step 50 | step 150 | step 225/300 |
|---|---|---|---|
| GRPO-V3 | 133 (0) | 153 (0) | **302 (0)** |
| DPO-V3 | 242 (0) | 185 (0) | 103 (0) |

**P1 — judge hits (deterministically-confirmed):**
| Method | step 50 | step 150 | step 225/300 |
|---|---|---|---|
| GRPO-V3 | 36 (9) | 51 (15) | 85 (14) |
| DPO-V3 | 48 (12) | 44 (10) | 31 (8) |

**GRPO escalates its S5 judge-hits 133→302 (2.3×) with *zero* genuine throughout**
— a measured reward-exploitation signature. GRPO's P1 genuine rate also falls
(25%→16%). DPO does not escalate (S5 hits decline), consistent with its
magnitude-invariant pairwise loss. This establishes that exploitation *occurred*
(for GRPO/S5), not merely that it was possible.

## 4. Corrected results — deterministic predicates only (judge removed)

`use_judge=False` in `TurnRewardAttributor`; reproduce with
`python scripts/rescore_deterministic.py`.

| Method | Coverage (judge) | **Coverage (deterministic)** | Yield/100 | Multi-pol |
|---|---|---|---|---|
| fixed (static, no oracle) | 20% | **5%** | 8 | 0% |
| scripted (static, oracle) | 60% | 55% | 118 | 8% |
| fewshot 30B | 50% | 45% | 122 | 12% |
| coop 30B | 65% | 60% | 162 | 22% |
| **GRPO-V3 s150** | 60% | **55%** | 216 | 22% |
| **DPO-V3 s300** | 60% | **55%** | 170 | 24% |
| GFlowNet-V3 s175 | 40% | 35% | 230 | 16% |

Only **S5** drops (everywhere) — its deterministic verifier never fires.
Rankings are preserved (GRPO=DPO=55%); the dedup ablation (16-policy-dominated)
is unaffected; and the **no-oracle gap widens**: static-without-oracle collapses
to **5%** while PROBE holds at **55%** (11×).

---

## 5. Draft author response

> We thank the reviewer for distinguishing verifier *self-consistency* from
> verifier *validity*. We agree that repeated agreement establishes
> reproducibility, not correctness — especially when the same verifier supplies
> both the training reward and the reported metric.
>
> We conducted a human evaluation of 208 flagged instances across 85 episodes
> with two independent annotators. Human–human agreement was 99.5% (Cohen's
> κ = 0.985), indicating the cases are unambiguous under our policy definitions.
> The LLM judge agreed with the human consensus on only 39.6% of instances
> (κ = 0.084), flagging 76.8% of turns as violations vs 20.3% under humans
> (121 false positives, 4 false negatives). The two High-severity semantic
> policies the reviewer highlighted are the worst: S5 precision 0% (0/40 flagged
> hits genuine; the judge penalises read-only *searches*, whereas S5 concerns
> booking/cancellation/modification), and P1 precision 18%.
>
> Rather than qualify a biased metric, we **remove the LLM judge entirely**: all
> four hybrid policies are now scored by their deterministic predicates, making
> the reward and the coverage metric fully objective (20/20 programmatic
> verifiers) and eliminating the reward-hackable component. Under this scoring
> the method comparison is unchanged — GRPO-V3 and DPO-V3 both move from 60% to
> 55% coverage (only S5 falls out, because its deterministic verifier never
> fires: the frozen target rarely reaches a mutation state, which reinforces our
> diagnosed coverage ceiling). The dedup ablation is unaffected, and the
> no-oracle result strengthens (a static suite without the answer key scores 5%
> vs PROBE's 55%). A cross-domain check on τ-bench, whose verifiers are entirely
> deterministic, shows the same ordering, corroborating that the effect is not a
> judge artifact.
>
> We further audit the auditor's hits across training checkpoints: GRPO's
> judge-flagged S5 hits escalate 133→302 while zero are deterministically
> confirmed, a measured signature of verifier exploitation; DPO does not exhibit
> this. We therefore narrow our claims — improvements are stated as
> *deterministically-verified* coverage, and we present the human study, the
> per-policy confusion matrices, and this diagnostic in the appendix.

## Things NOT to say (per ChatGPT's list — all correct)

- "The judge is reliable because it is self-consistent."
- "The human sample may be noisy." (κ=0.985 refutes this.)
- "High recall compensates for low precision."
- "The verifier is only used in training, so eval is unaffected."
- "A frozen verifier can't be gamed." (Freezing prevents drift, not gaming.)
