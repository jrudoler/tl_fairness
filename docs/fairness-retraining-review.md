# Data fairness, model fairness, and independent retraining

Review date: 2026-09-09. The manuscript edits are deliberately focused; this
document holds the implementation audit and the qualifications behind them.

## What the existing experiments establish

The parity simulations **already draw independent training datasets from a
fixed population**. The missing elements were a suitable parity null, explicit
rejection/false-negative rates, and a comparison with each fitted model's own
target. Holding evaluation *size* fixed in Experiment 5 does not mean reusing a
single evaluation dataset: that dataset is also redrawn in every replicate.

| Setting / implementation | Independent retraining? | Prediction 1: population disparity | Prediction 2: population null |
| --- | --- | --- | --- |
| Setting 1: `analysis/sim_parity`, `analysis/sim_tmle`, Experiment 1 | Yes | Partial: nonzero parity, point estimates/coverage or widths; no paired rejection analysis | No parity-null sweep |
| Setting 3: `analysis/sim_robust`, Experiment 2 | Yes | Partial: nonzero parity, bias/coverage under nuisance misspecification; bias and sampling variance are mixed | No parity-null sweep |
| Setting 3 training-size sweep: `analysis/sim_trainsize`, Experiment 5 | Yes; fresh training **and** evaluation observations | Closest existing comparison: fixed evaluation size, varying training size, data-target coverage; no own-model truth or paired false-negative rates | No |
| Setting 2 described in the manuscript | No executable implementation located in the current pipeline/experiment scripts | Not its purpose | Conceptual parity null, but `X` independent of `G` makes *every* fitted group-blind rule have zero population parity; it cannot demonstrate training-induced population model disparity |
| CMI simulations and Experiment 3 | Yes | Conditional-dependence alternatives are present | Conditional-independence null is present, but the global-permutation comparator targets marginal independence; this tests a conditioning mismatch, not omitted model-training variance |
| Conditioning-set experiment / Experiment 4 | Yes; DGP coefficients are fixed across replicates | Omitting confounders changes the CMI target | Full conditioning gives a CMI null, but there is no fixed-model parity comparison and ordinary Wald inference is nonregular at this null |
| New Experiment 6 / `sim_retraining` | Yes, with nested independent evaluations | Exact nonzero parity, coverage, power/FNR, paired rejections, and own-model truth | Exact regular parity null with group-associated features; Type-I error, coverage, and separate variance components |

Thus the existing parity experiments partially address (1), and the existing CMI
experiments contain nulls and alternatives for a different question. None was a
complete test of both proposed parity mechanisms.

## What needs qualification in the paper's logic

For training data `T`, distinguish the fixed population quantity

`Psi(P) = E[D_P(X) | G=1] - E[D_P(X) | G=0]`

from the training-dependent quantity

`theta_T(P) = E[m_T(X) | G=1] - E[m_T(X) | G=0]`.

A model interval can be correctly calibrated for `theta_T` and badly calibrated
for `Psi`. Under a data-fair null, its fitted model may really have a nonzero
population disparity. Calling that rejection a false positive is appropriate
**for the data-fairness question**, not for the fitted-model question.

The model estimator's unconditional variance decomposes as

`Var(estimate) = E_T[Var(estimate | T)] + Var_T(E[estimate | T])`.

Its conditional interval captures the first component. However, the TL estimator
has a different center and a different variance: its correction removes leading
sensitivity to nuisance error, rather than adding the second component of the
model estimator's variance to the first. The evaluation outcomes supply the
information for this correction. The comparison gives both methods the same
observations, but TL needs evaluation labels; a predictions-only audit cannot
implement this correction.

Wider intervals **cannot by themselves** reduce false negatives. With identical
centers, a wider symmetric interval rejects zero less often. Bias correction may
recover missed violations, but coverage, Type-I error, power, and paired
disagreements must be measured separately. Non-rejection is not certification
of fairness. A general superiority claim would also need size-calibrated power
comparisons, not just comparison with an anti-conservative test.

Nor is greater TL variance a universal theorem for arbitrary populations and
estimators. In the new design, `Y` and `G` are conditionally independent given
`X`, so at the true nuisances the score-only EIF and the residual correction are
uncorrelated. There the residual contributes a nonnegative extra variance.
That argument needs those conditions; the two terms need not be uncorrelated
in a general observed-data law.

The usual one-step theory also requires a small product remainder and an
adequate influence-function estimate. Double-robust *consistency* does not
guarantee valid Wald intervals when one nuisance remains misspecified, or when
training data are too small compared with evaluation data. These conditions are
central in [Hines et al., *Demystifying Statistical Learning Based on Efficient
Influence Functions*](https://arxiv.org/abs/2107.00681), already cited by the paper.

## Variance corrections made during the review

The manuscript's probabilistic group EIF is

`phi_g = [pi_g(X)(Y-D(X)) + I(G=g)(D(X)-Psi_g)] / P(G=g)`.

The implementations instead formed uncentered group contributions, subtracted
one overall contrast estimate, and took their variance. That omits the effect
of estimating each group denominator and can substantially inflate interval
width. The corrected variance centers each group contribution separately and
takes the variance of their difference on the original evaluation rows.

This is fixed in the shared one-step helper in `tlfair/metrics.py`, the duplicate
probabilistic-parity simulations in `tlfair/simulations.py`, and the NumPy and
JAX TMLE paths. The same denominator-centering correction applies to the
probabilistic equal-opportunity implementation. Point estimands and fitted
models are unchanged; intervals change. Tests check the one-step EIF against
finite differences of the empirical ratio functional, the fixed-score limit,
and the NumPy TMLE variances. The optional JAX equivalence test now checks SEs
as well as estimates, but JAX is not installed in this local environment.

The Figure 3 baseline also divided each group's prediction variance by the
full evaluation size instead of that group's size. This understated its
variance by approximately a factor of two for balanced groups. The corrected
Figure 3 was regenerated (seed 123, 50 repeats per size). At evaluation/training
size 1000, TL variance is `0.0006292` and model variance is `0.0005679`: about
11% more variance, or 5% more interval width. The old, much larger gap was not
clean evidence of accounted-for training uncertainty.

The subsequent full regeneration is documented in [the correction report](eif-regeneration.md).
It also found and corrected the thresholded-metric variance paths. Historical
CSV files and manuscript sections were archived before replacement; the report
records all changes in interpretation, including substantial coverage failures
and the changed Adult equal-opportunity conclusion.

## New experiment and reproduction

`experiments/exp6_retraining.py` uses `X=(U,V)` uniform on four cells,
`P(G=1|X)=(1+0.6U)/2`, and `P(Y=1|X)=expit(V+effect*U)`. Both labels are
Bernoulli draws, independent conditional on `X`. Effects `0`, `0.2`, and `0.6`
give exact parity gaps `0`, `0.04713018`, and `0.13999844` (rounded). The actual
CSV truth is computed by enumeration, not taken from these rounded values.
Saturated four-cell nuisance regressions contain the truth; weak half-count
smoothing handles empty cells. There is no optimizer randomness, feature
omission, or approximation restriction to confound training sampling noise.

The current default grid sweeps training sizes **100, 150, 225, 350, 500, 750,
1000, 1500, 2000** and evaluation sizes **250, 500, 1000, 2000, 4000**, independently,
for all three effects. Each of the 135 configurations has **1000 independent
training datasets and 50 independent evaluation datasets per fit**: 135,000
fits and 6,750,000 evaluation datasets.
Fifty is the number of evaluation *replicates*, not the number of observations
in an evaluation dataset. Methods share the same observations within a
replicate. Different size configurations use independent random streams.

Saved outputs include exact data and fitted-model truths; every estimate, SE,
interval and rejection; data- and own-target coverage; Type-I error/power/FNR;
paired disagreement rates; bias and width; and within/between-training variance
components. MCSEs use training-dataset clusters. Negative estimated between
components caused by Monte Carlo error are retained rather than hidden.
Raw rows stream to disk one configuration at a time, keeping memory bounded.

```bash
uv run --no-sync .venv/bin/python experiments/exp6_retraining.py \
    --train-sizes 100 150 225 350 500 750 1000 1500 2000 \
    --test-sizes 250 500 1000 2000 4000 \
    --reps 1000 --evaluations 50 --n-jobs 4
# Same run through the paper workflow, on the prescribed CPU partition:
sbatch slurm/retraining.sbatch
# Independence control (every group-blind rule has zero population parity):
uv run --no-sync .venv/bin/python experiments/exp6_retraining.py \
    --effects 0 --association 0 --output-dir experiments/out/retraining_independent_dense
```

The pipeline writes raw data, summaries, configuration and logs under
`data/generated/retraining_dense/`; its figure is `results/figures/fig9_retraining.pdf`.
The three PDF pages show coverage, rejection, and width, respectively, with
three horizontally arranged panels, one per population. Training size is on the horizontal axis,
evaluation size uses a continuous logarithmic color scale, and fairness method
is line style and marker shape. Each curve has nine simulated training sizes;
the intermediate points are new simulations, not interpolated estimates.
The known-regression ("Oracle score") control remains in the CSVs but is omitted
from these manuscript figures. Capped error bars and light bands show the actual
Monte Carlo uncertainty; very narrow errors are not artificially enlarged.
Companion PNGs are generated for inspection. Output paths are configurable
and anchored at the repository root, including when invoked from another
directory. Serial/parallel and streamed/in-memory equivalence are tested.

## Other manuscript issues to keep separate

- Setting 2 writes a group-dependent outcome regression as `P(Y=1|X)`. If the
  outcome depends on `G` given `X`, the group-blind regression is the mixture
  over `G|X`. Equal marginal means and `X` independent of `G` imply parity for a
  group-blind rule, but not equal opportunity after conditioning on `Y=1`.
- The deterministic legacy outcome mode does **not** preserve the same
  `D_P(X)=P(Y=1|X)` as a Bernoulli outcome law. If `Y=1{q(X)>1/2}`, the true
  regression is that indicator, not `q(X)`. The canonical pipeline now uses
  Bernoulli labels, but older default-off helpers/notebooks should not be
  interpreted as probabilistic-parity validation for the stated population.
- The thresholded derivation needs additional theoretical work: zero mass
  exactly at a threshold does not justify exchanging differentiation and
  integration. For example, `X~Uniform(-1,1)` and `D_t(X)=expit(X+t)` give
  `P(D_t(X)>=1/2)=(1+t)/2` near zero, with derivative `1/2`, despite
  `P(D_0(X)=1/2)=0`. This review does not validate the thresholded estimators or
  repair that separate derivation.
- The paper correctly notes a vanishing CMI EIF at conditional independence,
  but some conditioning-set passages still call the Wald test calibrated at
  that null. A negative Wald interval there is not positive evidence of
  independence. The parity experiment intentionally uses a regular null;
  solving CMI boundary inference is a separate task.

## Initial 3-by-3 results (archived)

The initial coarse sweep completed on whartonstat (Slurm job 80751), as did the matched
independence control (80752). This is 36,000 independent training fits and
1,800,000 evaluation datasets across the two designs. The table gives
population-null rejection rates and power for the smaller nonzero gap (0.04713).
All tests nominally use alpha=0.05; higher power of an anti-conservative model
test should not be interpreted as a comparison at equal actual test size.

| Training size | Evaluation size | TL Type-I error | Model Type-I error | TL power | Model power |
| --- | --- | --- | --- | --- | --- |
| 100 | 250 | 6.3% | 35.1% | 20.5% | 48.7% |
| 100 | 1000 | 10.1% | 61.1% | 55.4% | 71.6% |
| 100 | 4000 | 20.8% | 80.3% | 92.6% | 85.2% |
| 500 | 250 | 5.2% | 12.8% | 18.8% | 40.4% |
| 500 | 1000 | 5.5% | 32.5% | 56.2% | 75.1% |
| 500 | 4000 | 5.7% | 56.4% | 98.7% | 91.5% |
| 2000 | 250 | 5.1% | 7.2% | 18.8% | 36.9% |
| 2000 | 1000 | 5.1% | 13.0% | 56.4% | 83.6% |
| 2000 | 4000 | 5.0% | 30.8% | 98.8% | 99.0% |

At training size 500 and evaluation size 4000, the TL interval covers data
truth in 94.26% of null evaluations and 94.56% of alternative evaluations,
versus 43.58% and 41.60% for the model interval. The model interval still has
94.91% and 94.99% coverage of its **own** fitted-rule target. TL detects the
alternative when the model test does not in 8.25% of paired evaluations; the
reverse disagreement occurs in 1.10%. Overall false-negative rates are 1.32%
(TL) and 8.47% (model). Null rejection MCSEs are 0.12 and 1.15 percentage points;
alternative power MCSEs are 0.06 and 0.68 percentage points, respectively.

The variance decomposition supports the proposed mechanism in that same null
configuration. Model within-training variance is 0.000054, between-training
variance 0.000552, and mean reported variance 0.000054. For TL these are
0.000126, 0.000008, and 0.000125. Its mean interval is wider (0.0438 versus
0.0287), while its corrected center is much less sensitive to the training fit.

The larger grid supports **qualified versions of both predictions**: some
configurations show fewer false negatives and far fewer false positives, with
substantially better data-target coverage. However, TL can have lower power
at smaller evaluation sizes; with 100 training observations it also increasingly
under-covers as evaluation size grows. At 500 training observations the small
remaining excess null rejection (up to 5.74%) is detectable with these many
replicates, so the result is near-nominal rather than exact calibration.

The independence control is an essential counterexample to unconditional
superiority. Its model-fairness Type-I error stays between 4.97% and 5.19%
across the grid. TL reaches 38.17% at training size 100 / evaluation size 4000
and 7.64% at 500 / 4000, returning to about 5% at training size 2000. When every
fixed group-blind rule is already population-fair, estimating unnecessary
nuisances can worsen finite-sample calibration.

Validation: 29 local tests passed; the optional JAX module was skipped because
JAX is unavailable. The full expanded simulation and control both completed
on four-CPU compute-node allocations. Tests include numerical EIF derivatives,
interval regression checks, serial/parallel equivalence, streamed/in-memory
equivalence, and standalone parallel CLI execution from another directory.
The initial manuscript compiled successfully. Outside the simulation section, the
discussion, introduction, and inference section each have just one sentence
changed; the longer conceptual and implementation details remain in this review.

## Dense 9-by-5 sweep

The expanded sweep uses nine training sizes and five evaluation sizes at each of three population disparities. Each point retains 1000 independent training fits and 50 evaluations per fit. The main sweep has 135,000 training fits; the matched independence control adds 45,000. Figures 5 and 6 use three horizontal panels in a 5.85-by-2.7-inch figure.

The following results replace the archived coarse-sweep values above:

| Training size | Evaluation size | TL Type-I error | Model Type-I error | TL power | Model power |
| --- | --- | --- | --- | --- | --- |
| 100 | 250 | 6.3% | 35.1% | 20.0% | 46.0% |
| 100 | 500 | 7.5% | 48.6% | 33.2% | 59.2% |
| 100 | 1000 | 9.7% | 61.2% | 54.6% | 70.5% |
| 100 | 2000 | 13.8% | 70.1% | 78.3% | 78.4% |
| 100 | 4000 | 21.6% | 78.6% | 93.2% | 84.2% |
| 500 | 250 | 5.3% | 13.0% | 19.1% | 38.9% |
| 500 | 500 | 5.1% | 19.7% | 32.5% | 59.3% |
| 500 | 1000 | 5.4% | 31.6% | 56.2% | 74.7% |
| 500 | 2000 | 5.5% | 43.3% | 83.7% | 84.5% |
| 500 | 4000 | 5.8% | 55.9% | 98.6% | 90.7% |

At training size 500 and evaluation size 4000 under the smaller alternative, TL coverage is 94.14% versus 42.78% for model fairness, and power is 98.568% versus 90.736%. TL alone rejects in 8.974% of paired evaluations. Under the null at training size 100 and evaluation size 4000, TL rejection is 21.602%, so the small-training limitation remains. The independence control model rejection ranges from 4.858% to 5.344%, while TL reaches 36.464%. The conclusions remain qualified; neither method has uniformly superior power or calibration.

The dense run (Slurm 80756) reached the storage quota after 101 configurations. Completed raw rows were retained and compressed; job 80764 resumes configuration 102 with the original configuration-indexed seeds. The independence control completed as job 80757. Raw output now uses `replicates.csv.gz` to limit storage use, and the 14 retraining tests pass with compressed output.

The resumed sweep completed all 135 configurations. Together with the 45-configuration control, the dense experiments comprise 180,000 independent training fits and 9 million evaluation datasets.

Final validation: all 135 main and 45 control configurations have 1000 fits and 50 evaluations per fit. Compressed raw files contain exactly 20,250,000 and 6,750,000 rows, respectively. The manuscript compiled successfully; Figures 5 and 6 were visually checked on pages 13 and 14.
