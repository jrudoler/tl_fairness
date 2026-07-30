# Section 2 ("Inference for Data Fairness") — structural critique

Status: written 2026-07-30. Basis for the revision tracked in
[`section2-revision-plan.md`](section2-revision-plan.md).

Section 2 lives in `paper/sections/inference.tex` (121 lines, pp. 2–5 of the compiled
PDF). It is the methodological core of the paper: where the data-fairness estimands,
their efficient influence functions (EIFs), and their estimators are introduced.

## Why it reads as patchwork

The section has been through four structural surgeries with no re-plan in between:

| commit | change |
| --- | --- |
| `28346ae` | split the monolithic manuscript into per-section files |
| `00800ad` | "Tighten inference section; **drop flawed causal discussion subsection**" |
| `fb320ea` | "**Move Targeted Learning primer to appendix**; summarize TL in intro" |
| `ea78f83` | "**Add conditional mutual information** estimand, EIF, and estimator" |

Each moved material *out of* or *into* Section 2 without re-deriving what the section
needs to stand alone. Consequences, all still visible:

- the ATE framing expanded to fill the hole left by the dropped causal subsection;
- notation the primer used to define (`m(x)`, `\bar\Psi`, `\Prob_n`) now dangles;
- CMI was appended as a co-equal subsection without being integrated into the
  decision-rule story that §2.1 spends three pages building.

## Structure as it stood before revision

```
§2  Inference for Data Fairness
    ¶ intro: TL is flexible → define an estimand; "we take two approaches"
    §2.1  Fairness Metrics                                        [~3 pp., pp. 2–5]
        ¶ ATE when G randomized → mean difference
        ¶ ATE when G not randomized → g-formula, ignorability + positivity
        ¶ but G is immutable → counterfactual ill-defined → "treat the ATE with caution"
        ¶ decision-maker uses X to assess Y (employer/resume example)
        ¶ D(x) = P(Y=1|X=x); Bayes rule D_c(x) = 1{D(x) ≥ c}; two contrasts with ATE
        ¶ traditional DP is model-relative → replace ŷ with D_c
        ¶ standardized ATE vs. data-fairness estimand ("does the opposite")
        ¶ in practice estimate D with D̂ → TL; roadmap (DP here, EO → App. D)
        §2.1.1  Traditional Metrics       estimand (2)–(3), EIF (4), estimator (5)
        §2.1.2  Probabilistic Metrics     estimand (6), EIF (7) w/ π(x), estimator (8)
        §2.1.3  Double Robustness         claim + pointer to App. E; model-fairness
                                          contrast via m(x)
    §2.2  Conditional Mutual Information                          [~0.75 pp., p. 5]
        ¶ bridge: a mean difference can vanish while association persists
        ¶ estimand (9); non-negative; zero iff Y ⫫ G | X; "certifies conditional
          independence — a strong form of data fairness"
        ¶ EIF (10) = centered log density-ratio; degenerate at the boundary
        ¶ estimator (11): one joint model, marginalize, evaluate log-ratio;
          no targeting step needed; Wald interval
    % commented out: \subsection{Causal Explanation Formula}
```

Supporting material: App. A (TL primer + worked model-fairness example), App. B (CMI EIF
derivation), App. D (EO estimands/EIFs/estimators), App. E (DR proof), App. F (rate
conditions + positivity).

## The argument as written

1. TL lets you pick the estimand, so pick one describing the *data*, not a model.
2. The obvious data-level estimand — the ATE of group membership — is unattractive because
   the counterfactual "had this person been of another race" is ill-defined.
3. So reframe: decisions apply a rule to features. Model the rule as Bayes-optimal against
   the *true* `P(Y=1|X)`. This makes the rule a property of the DGP.
4. Swap the fitted `ŷ` for that `D_c` in the standard metrics. The result looks like the
   standardized ATE but is structurally its mirror image: `G` comes *out* of the
   regression, and averaging goes *over* the group-conditional law of `X`. The estimand
   measures the disparity a `G`-blind rule inherits from `X ⫫̸ G`.
5. `D` is unknown, so plugging in `D̂` biases the estimate; TL's EIF machinery corrects it
   and yields valid intervals.
6. Bonus: the probabilistic estimator is doubly robust — a property that does *not* arise
   for model fairness, because a fitted `m` is fixed w.r.t. perturbations of `P`.
7. A difference of means is coarse; it can be zero while `Y` and `G` remain associated
   given `X`. So add CMI, zero iff `Y ⫫ G | X`.
8. CMI's EIF is the centered integrand, so the plug-in is already efficient — but the EIF
   vanishes at conditional independence, so inference is non-regular exactly at the null of
   interest.

Steps 3–4 are the paper's best idea and are well written. The problems are around them.

---

## A. Major logic gaps

### A1. "CMI certifies a strong form of data fairness" is false, and §4 contradicts it

Line 106 claimed CMI, unlike DP/EO, "certifies *conditional independence* — a strong form
of data fairness." The bridge at line 102 argues only one direction (a mean difference can
vanish while association persists). The converse fails: `I(Y;G|X)=0` does **not** imply
`Ψ_DP = 0`, because parity is driven by `X ⫫̸ G`, about which conditional independence says
nothing. The estimands are logically **non-nested**, not ordered by strength.

Not academic: Table 1 in §4 reports, for *both* datasets, `CMI ≈ -0.00` alongside
`Parity = 0.17–0.19` with intervals bounded away from zero. A reader who believes line 106
reads that table as self-contradictory.

### A2. The thresholded EIF is asserted, and borrowed from a setting valid for another reason

Line 56 said "As in the example of Appendix A … Doing so, we find that the EIF is …". In
that example the integrand is `m_c(x)`, a **fixed fitted model**, functionally independent
of `P` — which is exactly why the pathwise derivative has only the `X|G` term. In §2.1.1
the integrand is `D_c(x) = 1{D_P(x) ≥ c}`, which **is** a functional of `P`. The derivative
therefore carries a second contribution,

```
E[ (d/dt) 1{D_{P_t}(X) ≥ c} | G=1 ],
```

which vanishes only under a **margin condition** (e.g. `P(D(X) = c) = 0`, and something
stronger for the plug-in bias to be second-order near the threshold). No such condition
appeared in §2 or App. F, whose positivity list covers `D(x), π(x) ∈ (0,1)` and marginal
group probabilities but never the threshold. As written, Eq. (4) was stated for the wrong
reason.

### A3. The paper's own naive-estimator critique applies to its own thresholded estimator

Eq. (5) has no nuisance-correction term; it is the mean of `D̂_c` over the `G=1` rows, i.e.
the group difference in thresholded predictions. `tlfair/metrics.py` says so in a comment:

> The EIF carries no outcome-residual term, so these are plug-in estimators.

Its EIF-based variance is the two-sample variance of binary predictions — essentially the
difference-in-means SE. But §3.2.1 ("A Model-Based Estimate") and §3.2.2 ("Does More
Training Data Rescue the Naive Estimator?") build the paper's strongest selling point on
the claim that treating `D̂` as truth under-covers and that more training data does not fix
it. Those simulations target the *probabilistic* metric, so the two claims do not literally
collide — but nothing said why the thresholded estimator is exempt, and the honest answer
(A2's margin condition, plus the absence of any correction term to add) is exactly the gap
in A2. A reader who connects §2.1.1 to §3.2.2 concludes the paper contradicts itself.

### A4. Only 2 of 5 estimands use TL's bias-correction step, and the layout hid it

- thresholded DP — plug-in
- thresholded EO — plug-in
- CMI — plug-in (line 118 stated this outright: "no separate targeting step is required")
- probabilistic DP, probabilistic EO — genuine one-step correction, and the only two with
  double robustness

This is a *finding*. The traditional/probabilistic/CMI split buried it, and line 118
presented it as a convenience rather than the same structural fact.

### A5. `Y` is treated as ground truth, with no caveat anywhere

Every estimand is a functional of `P(Y|X)` (or `p(G,Y|X)`). If `Y` is a recorded historical
decision or a biased proxy — bar passage, income, a prior hiring outcome — then "data
fairness" certifies fairness *relative to a possibly biased label*. EO makes this sharpest,
conditioning on `Y=1` ("among those who truly succeed"). Neither §2 nor §5's limitations
mentioned label bias or outcome measurement error. For a paper whose central move is
relocating fairness from the model to the data, this is the most conspicuous omission.

### A6. Double robustness is claimed in the wrong currency

§2.1.3 said DR means "we only need to estimate either `π` or `D` well for our estimator to
perform well," and App. E proves mean-unbiasedness when one nuisance is *exactly* correct.

1. Exact correctness is not the useful statement. The useful one is the **product-bias**
   form (the remainder is a product of the two nuisance errors), which is what licenses the
   `o_P(n^{-1/4})`-each rate requirement App. F invokes. The two appendices make adjacent
   arguments that were never connected.
2. What §3's Setting 3 demonstrates is **coverage** under misspecification, not
   point-estimate accuracy. §2.1.3 should promise interval validity, since that is what
   gets tested.

Also App. E concluded `|E[1/p̂] − 1/p| = 0`, false in finite samples by Jensen; it is an
asymptotic statement, and the preceding inequality additionally needs a sign argument.

### A7. The section named "Inference" contained no inferential claim

No asymptotic linearity, no `√n(Ψ̄ − Ψ) → N(0, Var φ)`, no variance estimator, no mention of
sample splitting, no numbered assumption or proposition. All of it lived in App. A and
App. F — i.e. *after* §3 has already started reporting Wald coverage. Every downstream
result depended on a theorem the main text never stated.

### A8. `D` is defined as `P(Y=1|X)` and then called a decision-maker's assessment

Lines 26–34 slid from "employers use `X` to judge whether an applicant will contribute" to
`D(x) = P(Y=1|X=x)` as *the* model of that assessment. That real decision-makers behave as
Bayes-optimal predictors of a well-defined `Y` is the load-bearing assumption of the whole
framework, asserted in one sentence. It deserves an honest paragraph: the estimand
describes the disparity a *well-calibrated, `G`-blind* decision-maker would produce — a
normative benchmark, not a description of any actual process.

## B. Parallelism failures

**B1. Equal opportunity is in the abstract, the intro, and nowhere in the main text.** The
abstract promises estimators for "demographic parity, equal opportunity, and conditional
mutual information"; line 44 exiled all of EO to App. D. Meanwhile CMI — one of the same
three — got a full main-text subsection. The reader could not see two of the three headline
estimands.

**B2. Depth is inverted relative to novelty.** ~3 pages on DP (a familiar metric),
including two full statements of the standardized ATE (lines 18 and 40) and two statements
of the immutability objection (line 22, already made at intro line 25), vs. ~0.75 pages on
the genuinely new estimand with the only non-standard inferential wrinkle in the paper.

**B3. Sectioning levels misrepresented the logic.** `§2.2 CMI` sat at the same level as
`§2.1 Fairness Metrics`, elevating CMI *above* DP and EO. `§2.1.3 Double Robustness` was
nested under "Fairness Metrics" though its content is a section-level data-fairness /
model-fairness contrast.

**B4. The internal template was not repeated.** §2.1.1 and §2.1.2 ran estimand → EIF →
estimator as numbered displays; §2.2 ran the same three steps in flowing prose with no
subsubsections.

**B5. Motivational scaffolding existed for one estimand only.** DP got the ATE contrast,
the decision-maker story, and the mirror-image argument. CMI got one sentence of bridge,
then arrived with an entirely different conceptual basis (information-theoretic conditional
independence) never referencing `D`, `D_c`, `c`, or a decision.

**B6. The nuisances were never presented as one object.** `D(x) = P(Y=1|X)` in §2.1;
`π(x) = P(G=1|X)` in §2.1.2; `ρ_g(x) = P(Y=1,G=g|X)` in App. D with no explanation of why
the nuisance changes form; `p(g,y|x)` in §2.2. But `p(g,y|x)` **subsumes all of them** —
`D`, `π`, `ρ_g` are margins or slices of the joint conditional law. Saying so once, up
front, unifies the section, explains EO's different-looking nuisance, and makes the CMI
estimator the same machinery rather than a new one.

**B7. No cross-estimand summary.** Five estimands × (nuisances, correction term?, DR?,
regular?, where derived). One table fixes B1, B4, B6, and A4 at once — the highest
value-per-line addition available.

**B8. Forward references were inconsistent.** Only the CMI paragraph (line 112) pointed to
§3 and §5. Traditional metrics, probabilistic metrics, and double robustness pointed at no
simulation, though Settings 1–3 exist precisely to test them.

## C. Material that felt irrelevant or misplaced

**C1. The ATE build-up ran about twice the length its conclusion supports.** Three
paragraphs to reach "we therefore treat the ATE with caution." The paper never estimates an
ATE and never reuses its identification conditions. The *contrast* (lines 40–42) is
valuable and stays; the setup compresses to one paragraph, especially since intro lines
19–25 already cover the immutability objection and the causal decision-fairness literature.

**C2. Notation used before definition** — all casualties of `fb320ea`:

| symbol | used in §2 | defined |
| --- | --- | --- |
| `m(x)` | lines 93–95 | App. A line 31 |
| `\bar\Psi(\hat P)` | Eq. (5) onward | App. A line 46 |
| `\Prob_n` | line 118 | App. F line 9 |
| "one-step estimator" | §3 lines 50, 57 | nowhere; App. A calls it "estimating equations" |

**C3. `c` means two different things.** In §2 it is the decision threshold
(`D_c(x) = 1{D(x) ≥ c}`, "typically `c = 1/2`"). In §3.2 and App. C it is the coupling
weight in the CMI simulation (`c S + U + f_β(X)`, swept 0 → 4, with a table captioned "for
each value of `c`"). Two unrelated meanings for one symbol in adjacent sections.

**C4. App. G (Mediation Analysis) was orphaned, unfinished, and compiled into the PDF.** No
`\label`, referenced nowhere, and line 16 read *"Giles derivation. Will add a little more
but basic result is …"*. Almost certainly residue of the causal subsection dropped in
`00800ad`. It shipped.

**C5. Two commented-out causal placeholders** — `inference.tex:120`
(`\subsection{Causal Explanation Formula}`) and `discussion.tex:9–13`
(`\subsection{Causal Inference for Data Fairness}`) — left the thread's status ambiguous.

**C6. A word was broken across a source newline.** `the conditional associati` /
`on between …` rendered as **"associati on"**. This turned out to be an *uncommitted*
regression in the working tree; the committed text was correct. (Separately, intro line 42
read "Because we define estimand is defined directly on the data-generating process".)

## D. Smaller correctness and consistency items

- **D1.** `Ψ_DP` subscripted for thresholded DP (Eq. 3) but probabilistic DP plain `Ψ(P)`
  (Eq. 6), making the two notationally indistinguishable; `Ψ_1` and `φ_1` each reused for
  both.
- **D2.** App. D Eqs. (12) and (14) center on `Ψ(P)` where it should be `Ψ_1(P)` — the
  one-group component, as in the corresponding §2 equations. Straight error.
- **D3.** App. D Eq. (13), the thresholded-EO estimator, plugs in `\hat D(x_i)` where the
  estimand and EIF use `D_c(x)`. Missing subscript.
- **D4.** `n*\hat\Prob(G=1)` (Eqs. 5, 8, App. D) uses `*` as multiplication in display
  math.
- **D5.** App. A line 50's Wald interval reads `\bar\Psi ± (1.96/\sqrt n) Σ_i φ_i²` —
  missing the `1/n` and the square root. §2 and §3 have it right.
- **D6.** `c = 1/2` called "typical" without noting it is Bayes-optimal under symmetric
  loss; nothing on how the thresholded verdict moves with `c` — the first knob a
  practitioner reaches for.
- **D7.** §2 states CMI is non-negative; Table 1 reports `-0.00 (-0.00, -0.00)` for both
  datasets. The plug-in is *not* a non-negative estimator, and near the boundary its
  interval can sit entirely below zero.
- **D8.** The "conditional association" framing promises a measure of association between
  `G` and `Y`; the estimand delivered is symmetric in `G` and `Y`, worth a clause since
  every other estimand in the paper is directional.
