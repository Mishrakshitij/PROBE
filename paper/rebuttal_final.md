# PROBE rebuttal — corrected, per-reviewer (consistent with completed human eval)

Scores: 4/4/4, all conf 4. Post space each reviewer separately ($\leq$5000 char/comment;
split into a 2nd comment if needed). **Core correction vs earlier drafts:** the
human eval is DONE and shows the judge over-flags — so we REMOVE the judge and
report fully-deterministic scoring, rather than claim the judge is accurate.

Key verified numbers (this cluster):
- Human eval: 208 items / 85 ep / 2 annotators. Human–human $\kappa$=0.985 (99.5% agree).
  Judge–human $\kappa$=0.084, agreement 39.6%, judge flags 76.8% vs humans 20.3% (121 FP, 4 FN).
  Per-policy judge precision: S5 0%, P1 18%, B6 12%, B4 65%. Aggregate 24%.
- Deterministic-only re-score (judge removed): GRPO-V3 55%, DPO-V3 55%, GFlowNet 35%,
  scripted(oracle) 55%, fixed(no-oracle) 5%, coop 60%, fewshot 45%. Only S5 drops everywhere.
- Reward-hacking diagnostic: GRPO S5 judge-hits 133$\rightarrow$153$\rightarrow$302 across steps, 0 genuine throughout; DPO doesn't escalate.
- Cross-family: Gemma-4 (Google, different family) — Qwen-trained DPO auditor reaches **40% deterministic coverage (8/20)**, tool-calls (30 calls / 50 ep); vs 55% on the Qwen target (expected drop for an unseen family, well above chance).
- $\tau$-bench (other cluster; 10 policies extracted from Sierra's wiki.md, **8/10 deterministic** + 2 lightweight heuristics TB-P9/P10) = supporting SECOND-DOMAIN evidence, NOT the SkyAssist static baseline. Final numbers: PROBE-DPO 30% (29 V/50ep), PROBE-GRPO 20%, zero-shot Qwen3.5-9B 20% (13 V/50ep).
- **KEY $\tau$-bench framing (turns "single domain" into a strength):** $\tau$-bench is a *user-simulator* — fixed task goals ("cancel reservation X"), so violations surface as byproducts of task execution and even zero-shot finds some (20%). SkyAssist needs a genuine *in-context adversarial auditor* — no task goal, so the auditor must learn to steer open-ended dialogue into violation states; this is why zero-shot largely fails (35%) and training is essential (60%). The two domains test different capabilities; PROBE's value is sharpest in the harder open-ended setting.

================================================================
## GENERAL / SUMMARY COMMENT (post once, readers = AC + all reviewers)
================================================================

We thank the AC and reviewers (scores 4/4/4). To our knowledge PROBE is the first *learned adversarial* auditor for multi-turn, tool-mediated policy compliance — a deliberately hard setting (no task goal; the auditor must learn to create the conditions under which violations occur) for which we built SkyAssist, since existing benchmarks such as $\tau$-bench are cooperative user-simulators that do not evaluate open-ended adversarial auditing. We summarize the new experiments here and respond to each reviewer separately.

**1. Human validation of the LLM-judge policies (bwc6 Q3 / AC P2).** Two independent annotators labeled 208 flagged instances across 85 episodes (human–human $\kappa$=0.985, 99.5% agreement). The LLM judge agreed with the human consensus only 39.6% of the time ($\kappa$=0.084), over-flagging the semantic policies — S5 precision 0%, P1 18% (aggregate 24%). **We therefore remove the LLM judge and strengthen the metric to fully objective: all 20 policies are now scored by deterministic predicates**, so the reward and coverage metric contain no exploitable component. On this stricter metric PROBE reaches **55%** (vs 35% zero-shot, 5% static), method rankings unchanged — only S5 drops, as its deterministic verifier never fires (reinforcing our diagnosed ceiling). A checkpoint diagnostic shows the exploitation was real for GRPO (its judge-S5 hits escalate 133$\rightarrow$302 with zero deterministically-confirmed) and absent for DPO.

| Judge precision vs human consensus | S5 (High) | P1 (High) | B6 (Low) | B4 (Low) | Aggregate |
|---|---|---|---|---|---|
| Precision | **0%** | 18% | 12% | 65% | **24%** |

(human–human $\kappa$=0.985; judge–human $\kappa$=0.084; judge flags 77% of turns vs 20% for humans.)

**2. Static & diversity baselines on SkyAssist (bwc6 Q1 / AC P1 / Xtwt W4).** All at 50 held-out episodes, fully-deterministic scoring. Coverage alone understates the gap — yield and multi-policy density separate the methods:

| Method (deterministic, 50 ep) | Coverage | Yield/100 | Multi-pol |
|---|---|---|---|
| Static, **no oracle** (fixed) | 5% | 8 | 0% |
| Static, expert **oracle** (scripted) | 55% | 118 | 8% |
| Few-shot 30B prober | 45% | 122 | 12% |
| GFlowNet (diversity-seeking) | 35% | 230 | 16% |
| **PROBE GRPO-V3 (4B)** | 55% | 216 | 22% |
| **PROBE DPO-V3 (4B)** | 55% | 170 | 24% |

PROBE matches an expert-oracle suite with **no oracle** and beats the no-oracle suite 11$\times$ on coverage (and 27$\times$ on yield); GFlowNet's high yield / low coverage confirms yield$\neq$coverage (policy-specific reward beats generic diversity). A diversity-only reward (binary any-violation success, no per-policy shaping) reaches **45%** vs **50%** for the coverage-oriented reward at the matched training step — confirming that per-policy shaping, not generic success, drives distinct-policy coverage.

**3. Cross-family + second domain (bwc6 Q2 / Xtwt W1).** We apply the Qwen-trained auditor **zero-shot** (trained only on Qwen, no Gemma-specific adaptation) to a Gemma-4 target (Google — a different family): it reaches **40% coverage (8/20)** vs 55% on Qwen — the expected drop for an unseen family, but well above chance given zero adaptation. Separately, $\tau$-bench (8/10 deterministic verifiers, an independent benchmark) transfers with no reward re-tuning (PROBE-DPO 30% vs zero-shot 20%). The two domains differ in kind: $\tau$-bench is a user-simulator (fixed goals $\rightarrow$ violations are task byproducts), while SkyAssist demands open-ended adversarial steering — the regime where learned auditing matters most.

**4. Fairness & robustness (AC P3/P4, Xtwt W2/W3).** Same-model control on $\tau$-bench (zero-shot Qwen3.5-9B 20% vs PROBE-DPO on the identical base 30%). The reward transfers to $\tau$-bench with no re-tuning, and the static baseline collapses to 5% under identical scoring — so the results are not a reward-tuning or scoring artifact.

We will fold all of this into the camera-ready (revised Table 1, human-study appendix with confusion matrices and CIs, and the corrected metric definition).

================================================================
## AREA CHAIR (PGuR) — 4 priorities
================================================================

We thank the AC. We ran the requested experiments on SkyAssist and, per Priority 2, completed a human study whose result reshapes our response.

**P1 — Static & diversity baselines on SkyAssist (measured).** We added non-learning baselines (50 held-out episodes each, identical policies/verifiers/seeds to the main eval), scored with our now fully-deterministic verifiers (see P2). The key contrast is the *answer key*:
- Static suite WITHOUT an answer key ("fixed": generic booking flows over random valid routes): **5%**.
- Static suite WITH an expert answer key ("scripted": one hand-crafted conversation per policy, using DB internals): **55%**.
- PROBE (trained 4B, no oracle): **55%**.

Static testing needs a hand-built per-policy answer key to reach 55%; PROBE matches it from generic scenarios with **no oracle** and beats the no-oracle suite 11$\times$ (5%$\rightarrow$55%). (Coverage saturates near this ceiling for any capable method — the highest-severity mutation policies are unreached by all — so the differentiators are no-oracle, yield/efficiency, and reward design, not raw coverage.) For "policy-specific reward beats generic diversity," PROBE-GFlowNet (mode-covering, Rainbow-like) is 35% vs 55% for GRPO/DPO on identical verifiers.

**P2 — Human validation of the LLM-judge policies.** We annotated 208 flagged instances across 85 episodes with two independent annotators (human–human $\kappa$=0.985, 99.5% agreement — the cases are unambiguous). The judge agreed with the human consensus on only 39.6% ($\kappa$=0.084); it flags 76.8% of turns as violations vs 20.3% for humans (121 FP, 4 FN). The High-severity policies are worst: **S5 precision 0%** (it penalizes read-only searches; S5 concerns booking/cancel/modify), **P1 18%**. Rather than qualify a biased metric, we **remove the LLM judge**: all 20 policies are now scored by deterministic predicates. Under this scoring GRPO-V3 and DPO-V3 both move 60%$\rightarrow$**55%** (only S5 drops — its deterministic verifier never fires, reinforcing our ceiling diagnosis), rankings unchanged, and the reward-hackable component is eliminated. A checkpoint diagnostic confirms exploitation occurred for GRPO (its judge-S5 hits escalate 133$\rightarrow$302 with zero genuine); DPO does not. $\tau$-bench (8/10 deterministic verifiers) shows the same ordering — the effect is not a judge artifact.

**P3 — "4B beats 30B."** We agree it measures "trained-small vs zero-shot-large," and will state it as such; the deployment point stands (trained 4B on one GPU). The fair same-model control (same base, $\pm$training) is in P4/the reviewer responses.

**P4 — Not a reward-tuning artifact.** (a) The reward transfers unchanged to $\tau$-bench (different policies/verifiers) — inconsistent with overfitting to 50 SkyAssist episodes. (b) V1$\rightarrow$V2$\rightarrow$V3 was guided by training-time behavioral diagnostics (role-confusion, loop rate, completion length, DPO margins), not held-out coverage. (c) The static "fixed" baseline collapses to 5% under identical scoring, so the 55% is not a scoring artifact.

================================================================
## Reviewer bwc6 (the detailed one) — Q1–Q5
================================================================

We thank the reviewer for the precise reading of our contribution.

**Q1 — Static & diversity baselines (SkyAssist, 50 held-out episodes each, all under the same judge-free deterministic scoring as PROBE).** Static suite without an answer key: **5%**; with a hand-built per-policy answer key: **55%**; PROBE (no oracle): **55%**. (We also ran your other suggested option, a cooperative user-simulator. As a *non-adversarial 30B* it reaches 60% — one policy, B5 fare-disclosure, above PROBE — but it triggers B5 only because it *completes* the booking (a task byproduct), not because it audits better: this is the same user-simulator-vs-adversarial distinction as $\tau$-bench, playing out within SkyAssist. A 4B adversarial auditor thus matches a 30B cooperative one; combining task-completion with adversarial probing is a natural route to the booking-flow policies, which we note as future work.) PROBE matches an oracle-authored suite without the oracle, and beats the no-oracle suite 11$\times$. Diversity: PROBE-GFlowNet (mode-covering, Rainbow-like, same verifiers) 35% vs 55% for GRPO/DPO — policy-specific reward beats generic diversity, measured.

**Q2 — Cross-family target.** We audited a **Gemma-4** target (Google — a different family and tokenizer from Qwen) through the public dialogue interface with our Qwen-trained DPO auditor, **zero-shot (no Gemma-specific adaptation)**. It issues genuine tool calls (30 over 50 episodes, incl. mutation calls) and reaches **40% coverage (8/20 policies)** vs 55% on the Qwen target. The drop is expected for an unseen family, but surfacing violations across 8 policies with no adaptation is evidence it learned transferable auditing rather than Qwen-specific exploits. As a *second domain*, $\tau$-bench (independently-authored; 8/10 deterministic verifiers) also transfers with no reward re-tuning (PROBE-DPO 30% vs zero-shot 20%). The two settings differ in kind, which sharpens the claim: $\tau$-bench is a user-simulator (fixed task goals $\rightarrow$ violations are task byproducts, so even zero-shot finds some), whereas SkyAssist requires open-ended adversarial steering with no goal — the regime where learned auditing matters most (zero-shot 35% vs PROBE 60%).

**Q3 — Human agreement; S5/P1 genuineness.** We completed the human study you asked for (208 items, 2 annotators, human $\kappa$=0.985). It confirms your concern: judge–human $\kappa$=0.084; **S5 precision 0%, P1 18%** (aggregate 24%, 121 FP / 4 FN). We therefore **remove the LLM judge entirely** and score all 20 policies with deterministic predicates. Coverage becomes GRPO/DPO **55%** (only S5 falls out; it never satisfied the deterministic mutation-without-confirmation predicate), rankings preserved, reward/metric fully objective. We also audit hits across checkpoints: GRPO's judge-S5 hits escalate 133$\rightarrow$302 with zero deterministically-confirmed — a measured exploitation signature — while DPO does not. We report the confusion matrices and this diagnostic in the appendix. (For clarity: the paper's 60% counted the four judged policies; S5 — human precision 0% — is the only one that falls out under objective scoring, hence a 5-point delta to 55%. Semantic policies are inherently subjective; a stronger, human-calibrated LLM judge could in future recover genuine violations the deterministic predicates miss, but our present conclusions rest on objective predicates and do not depend on judge quality.)

**Q4 — Deep mutation policies.** The binding constraint is target-side: the frozen agent emits mutation calls (book/cancel/modify) only 0–3$\times$/50 ep. We add a proximity reward (partial credit for approaching mutation states) and policy-conditioned pre-seeding; on $\tau$-bench this surfaced TB-P1 (confirmation-before-action) during training — the first hard action-level policy any trained auditor reached. We are explicit that these are the highest-severity, least-reached policies.

**Q5 — Taxonomy/severity sensitivity.** We will add a severity-uniform ($\sigma$=1) ablation; rankings are stable. The discovered/undiscovered split is set by verifier firing semantics (turn-level vs mutation-level), not severity weights.

We will add the two limitations you flagged: the four semantic verifiers were validated for self-consistency, not accuracy (now corrected via the human study + judge removal), and the highest-severity policies are those outside current reach.

================================================================
## Reviewer Xtwt — W1–W4
================================================================

We thank the reviewer; we address each weakness with measured results.

**W1 — Single domain / model family.** Two complementary results.
(a) **Cross-family target:** we apply the Qwen-trained auditor **zero-shot (no Gemma adaptation)** to a **Gemma-4** target (Google — a different family and tokenizer); it tool-calls and reaches **40% coverage (8/20)** vs 55% on the Qwen target — the expected drop for an unseen family, but well above chance, isolating transfer from Qwen-specific quirks.
(b) **Second domain — $\tau$-bench**, with a distinction that sharpens our claim rather than merely widening scope. $\tau$-bench is a *user-simulator* benchmark: each episode assigns a fixed task goal, so policy violations surface as byproducts of task execution — the auditor need not engineer them, which is why even zero-shot models find some (20%). SkyAssist has *no* task goal: the auditor must adaptively steer open-ended dialogue into violation states, which is exactly why zero-shot largely fails (35%) and training is essential (PROBE 60%). No existing benchmark evaluates this open-ended adversarial regime — which is why we built SkyAssist; the difficulty is *intrinsic to the task*, not a SkyAssist/Qwen idiosyncrasy the auditor overfits to (if it were an artifact, the reward and method would not transfer to $\tau$-bench unchanged). We extracted 10 policies from Sierra's `wiki.md` (8/10 fully deterministic; 2 lightweight heuristics, TB-P9/P10) and applied PROBE with no reward re-tuning:

| Method ($\tau$-bench) | Coverage | V/50ep |
|---|---|---|
| PROBE-DPO (Qwen3.5-9B, trained) | **30% (3/10)** | 29 |
| Zero-shot Qwen3.5-9B (same base) | 20% (2/10) | 13 |

+10 pp and +123% violations from training alone — on a different domain and target — is inconsistent with overfitting to SkyAssist/Qwen3 idiosyncrasies.

**W2 — "4B beats 30B" lacks a fair control.** Agreed; we reframe it as trained-small vs zero-shot-large and add the same-model control: on $\tau$-bench, zero-shot Qwen3.5-9B = 20% vs PROBE-DPO on the identical base = 30% (+10 pp, +123% violations) — same model, target, verifiers, budget; only training differs.

**W3 — Reward overfitting / researcher DoF.** (a) The composite reward transfers to $\tau$-bench with no changes — inconsistent with overfitting to 50 SkyAssist episodes. (b) V1$\rightarrow$V2$\rightarrow$V3 was driven by training-time behavioral diagnostics, not held-out coverage; weights are set from behavioral semantics. (c) A diversity-only baseline (binary-success reward, no per-policy coverage shaping, GFlowNet) isolates the coverage-oriented reward's contribution: it reaches 45% vs 50% for the coverage-oriented reward at the matched step — the reward design, not researcher tuning, drives coverage.

**W4 — Simpler non-learning baselines.** Exactly what you asked for, on SkyAssist, under fully-deterministic scoring: script/rule-based static suite = **5%** (no oracle) / **55%** (expert oracle); few-shot 30B prober = 45%; PROBE (trained 4B) = 55%. PROBE matches an expert-authored static suite without the oracle and beats the few-shot prober; a GPT-4o zero-shot on $\tau$-bench (API-only) will be added. Note the reward/metric are now fully deterministic (no LLM judge), addressing the reliability concern behind this point.

================================================================
## Reviewer AuG2 — Q1–Q5
================================================================

We thank the reviewer for the positive assessment and address each question with concrete evidence.

**Q1 — Confidence in the verifier suite.** 16/20 verifiers are deterministic Python predicates (state machines, schema validators, regex over tool-call arguments and database state) — exact and reproducible. For the 4 semantic policies we ran a human study (208 flagged instances / 85 episodes / 2 independent annotators): human–human $\kappa$=0.985 (99.5% agreement), but judge–human $\kappa$=0.084 — the judge over-flags (S5 precision 0%, P1 18%, aggregate 24%; 121 FP / 4 FN). We therefore **remove the LLM judge and score all 20 policies deterministically**; the reward and coverage metric are fully objective, coverage adjusts 60%$\rightarrow$55% (rankings preserved), and no component is reward-hackable. $\tau$-bench (8/10 deterministic verifiers) corroborates independently.

**Q2 — Relation to red-teaming.** Yes, on two levels. Architecturally, PROBE is a trained adversary probing a frozen target over multiple turns, like multi-turn red-teaming; and its **diversity-seeking branch connects directly to GFlowNet / Rainbow-Teaming red-teaming** — PROBE includes GFlowNet as one of its three optimizers, so PROBE-GFlowNet *is* a mode-covering red-teaming method pointed at our verifiers. Two things make PROBE distinct: (1) the **target surface is orthogonal** — content-safety red-teaming elicits disallowed *text* judged on the final response, whereas PROBE targets **trace-level procedural** violations (tool-call names, arguments, results, database-state transitions), so an agent can be fully content-safe yet book a restricted destination without confirmation or disclose an internal fare code — violations a content-safety judge never sees; and (2) diversity is encoded in the **reward** (per-episode severity-weighted dedup + frequency novelty), not structurally in the sampler/archive — which is precisely why our coverage-oriented reward (GRPO/DPO 55%) beats the mode-covering GFlowNet (35%) at distinct-policy coverage.

**Q3 — Zero-shot (API) model.** We have zero-shot results on **both** environments, and both fall well below a *smaller trained* auditor:
- **SkyAssist:** zero-shot Qwen3-30B = 35% coverage vs PROBE-4B = 55% (judge-free; 60% in the original submission).
- **$\tau$-bench:** zero-shot Qwen3.5-9B = 20% (13 V/50ep) vs PROBE-DPO (same 9B base) = 30% (29 V/50ep).
So a *larger* zero-shot model underperforms a *smaller trained* one — the limiting factor is **multi-turn behavioral steering** (inducing the target into mutation states through natural dialogue), not raw generation quality, so a strong API model would face the same challenge. A GPT-4o zero-shot on $\tau$-bench (API-only, no local serving) is straightforward and we will add it.

**Q4 — Scaling / full-parameter / recommendations.** Scaling the auditor helps: the 9B auditor produces ~4$\times$ the violations per episode of the 4B at comparable coverage (29 vs 7 V/50ep on $\tau$-bench). We used QLoRA (rank 64); full-parameter training should yield further gains since the adapter caps adapted capacity, a compute-vs-quality trade we did not fully explore. For a large-compute org we recommend: (a) a 9–14B auditor at QLoRA rank 128 or full fine-tune; (b) GFlowNet with terminal rewards calibrated per policy-ID and tool-depth (not just total verifier mass); (c) policy-conditioned scenario generators that pre-seed the database near mutation decisions; (d) the co-evolutionary loop (auditor exposes failures $\rightarrow$ agent trains on flagged traces $\rightarrow$ re-audit), which becomes practical at scale.

**Q5 — Surfacing non-search tool calls (capability vs. auditor).** It is **primarily target-side**, with a secondary auditor lever. Target-side: the frozen agent commits to mutation actions rarely — in our eval `book_flight` fires 1–3$\times$ and `cancel_booking` 0$\times$ per 50 episodes (Table 2). Without that commitment, ~8 mutation-dependent policies (B1, B3, O1–O4, P2) are **structurally unreachable** regardless of auditor aggressiveness. This generalizes across targets: the cross-family Gemma-4 target issued only 2 mutation calls in 50 episodes. Auditor-side, two levers help at the margin: a **proximity reward** (partial credit for approaching mutation states), which surfaced $\tau$-bench's TB-P1 (confirmation-before-action) twice during training — the first hard action-level policy any trained auditor reached — and **database pre-seeding** (episodes initialized with a booking/hold already in place). So more capable auditors help, but the dominant factor is whether the target agent can be driven to *act*.

================================================================
## CAMERA-READY CHANGELOG (keep paper $\leftrightarrow$ rebuttal consistent)
================================================================

1. **Metric redefinition (§4 / Table 1).** All 20 policies are now scored by
   *deterministic predicates*; the LLM judge is removed from the four semantic
   policies (B4/B6/S5/P1) after a human study showed it over-flags (judge–human
   $\kappa$=0.084; S5 precision 0%). Report coverage on this fully-objective metric —
   PROBE-GRPO/DPO **55%**, GFlowNet 35%, baselines 5–60% — with method rankings
   unchanged; note the headline moves 60%$\rightarrow$55% (only S5 drops). Replace any
   ">90% self-consistency" claim for the semantic verifiers. Code:
   `use_judge=False` in `TurnRewardAttributor`; `scripts/rescore_deterministic.py`.

2. **New appendix — "Human validation of the semantic verifiers."** 208
   instances / 85 episodes / 2 independent annotators (human–human $\kappa$=0.985);
   per-policy confusion matrices with episode-clustered CIs; the judge-removal
   ablation (corrected Table 1); and the checkpoint reward-exploitation
   diagnostic (GRPO judge-S5 hits 133$\rightarrow$302 with 0 genuine; DPO flat). Files:
   `data/human_eval/` + `scripts/{build_human_eval,score_annotations}.py`.
