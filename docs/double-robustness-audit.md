# Double robustness: proof and experiment audit

The product-of-errors identity for probabilistic parity is correct. The decline
in coverage after correcting the variance does not refute double robustness of
the point estimator. The previous experiment treated a fitted boosting model as
"correct" and interpreted coverage as a test of double robustness. Neither step
is justified without further conditions.

## What changed in the existing experiment

The paired rerun keeps each dataset, nuisance fit, and point estimate identical
and changes only the variance calculation. At total sample size 2,500 (1,250
training and 1,250 evaluation observations), the results are:

| Nuisances | Mean bias | Empirical SD | Mean corrected SE | Old coverage | Corrected coverage |
|---|---:|---:|---:|---:|---:|
| Boosted outcome, linear group | -0.03438 | 0.02266 | 0.01943 | 96% | 52% |
| Linear outcome, boosted group | -0.02981 | 0.02226 | 0.01733 | 94% | 59% |
| Both boosted | -0.00778 | 0.02373 | 0.02272 | 100% | 94% |

These are 100 paired replicates, from
`experiments/out/eif_regeneration/robust_paired_replicates.csv`. The corrected
half-width in the first row is about 0.038, only slightly larger than its bias.
The old, inflated intervals concealed the remaining estimation error.

Both true nuisance functions are logistic in squared features. The linear
models use raw features and are misspecified. Default gradient boosting is a
flexible approximation, but its finite-sample prediction error is not zero;
the experiment establishes neither consistency nor the required convergence
rates for its fixed tuning. Using a flexible learner is not equivalent to
supplying the true nuisance or fitting the correctly specified quadratic GLM.

## Exact decomposition and what the proof establishes

Condition on the training data. Let $P$ denote expectation over a fresh
observation, $P_n$ the evaluation average, $I=1\{G=1\}$, $p=P I$, and
$\widehat p=P_n I$. Define

$$A=\widehat\pi(X)(Y-\widehat D(X))+I\widehat D(X),\qquad
C=P[(\widehat\pi-\pi)(\widehat D-D)].$$

The manuscript's algebra gives the exact identity

$$PA=p\Psi_1-C.$$

For the implemented ratio estimator, on the event $\widehat p>0$, the exact
sample decomposition is

$$\widehat\Psi_1-\Psi_1=
\frac{-C+(P_n-P)(A-\Psi_1 I)}{\widehat p}.$$

This proves consistency when either nuisance is L2-consistent and the other
is bounded, with evaluation size also increasing and $p$ bounded away from
zero. The probability predictions here are bounded in [0,1]. No conditional
independence of Y and G given X is needed for this identity.

For the contrast between groups 1 and 0, the conditional population drift is

$$-C\left(\frac1p+\frac1{1-p}\right),$$

because the error in the group-0 propensity is the negative of the error in
the group-1 propensity. If either nuisance is supplied exactly, this drift
vanishes identically.

The empirical denominator adds a finite-sample ratio bias; an exact zero
product does not imply exact finite-sample unbiasedness. The appendix's passage
from the numerator identity to the expectation of the ratio should explicitly
condition on training and justify the ratio expansion. With bounded scores,
fixed positive p, and appropriate treatment of the exponentially rare empty
stratum event, that expectation correction is O(1/n). This is a rigor and
notation issue, not the explanation for the large boosting biases.

## Why consistency does not suffice for the reported intervals

Consistency needs $C=o_P(1)$. Negligible drift at the evaluation standard-error
scale needs the stronger $C=o_P(n_{eval}^{-1/2})$. The familiar sufficient bound
is

$$\|\widehat\pi-\pi\|_2\|\widehat D-D\|_2
=o_P(n_{eval}^{-1/2}).$$

When one nuisance remains misspecified, its error need not shrink. Even if the
other is estimated at the parametric $n_{train}^{-1/2}$ rate, the product can
remain of that order. With comparable training and evaluation sizes, the
training-fit drift can contribute first-order randomness. Its mean can be
nearly zero across training samples while its variance remains consequential.
The variance of the evaluation scores alone does not generally include that
extra training contribution.

The paper's asymptotic-linearity proposition also explicitly requires
$\|\phi(\widehat P)-\phi(P)\|_2=o_P(1)$. Sample splitting does not guarantee
this. Under one misspecified nuisance the score generally has a different
limit, so the efficient-IF proposition cannot automatically be invoked, even
when the product drift happens to vanish. With one exact nuisance, a separate
CLT for that limiting, generally inefficient score can still justify the
empirical-score variance. The exact-probability controls below check this
case directly.

The relevant manuscript overstatements are in `paper/sections/inference.tex`:
sample splitting alone making the empirical-process term negligible, and the
claim that the remainder condition is the only hypothesis not automatic from
construction. The proposition itself includes the extra score-consistency
assumption. The core double-robustness algebra does not need to be discarded.

This distinction is established in the literature, e.g. Benkeser, Carone,
van der Laan and Gilbert (2017),
[Doubly robust nonparametric inference on the average treatment effect](https://ctml.berkeley.edu/node/81),
and Dukes, Vansteelandt and Whitney (2024),
[On Doubly Robust Inference for Double Machine Learning in Semiparametric Regression](https://jmlr.org/papers/v25/22-1233.html).
These works concern related estimands; the decomposition above is specific to
this paper's estimator.

## Diagnostic controls

`analysis/eif_regeneration/robust_controls.py` retains Setting 3's Bernoulli
outcome and group laws and the corrected estimator. It independently redraws
training and evaluation samples in every replicate. It compares the actual
generating probabilities, quadratic logistic regressions that contain the
generating laws, and deliberately misspecified linear logistic regressions.
It fits each learned nuisance on training data only.

Results and individual replicates are saved in
`experiments/out/eif_regeneration/dr_controls/`. Population truth is integrated
using eight independent scrambled Sobol sequences, integrating out the
Bernoulli group noise. Its estimated integration standard error is 0.000052,
negligible compared with estimator standard errors around 0.02. The earlier
experiment used an independent Monte Carlo truth; its difference of about
0.00015 does not explain the biases of 0.03.

The diagnostic job uses 5,000 repetitions at each of 1,250 and 12,500 training
observations, with 1,250 evaluation observations throughout. The earlier
500-repetition pilot was expanded to distinguish modest undercoverage from
Monte Carlo error. Slurm job 80800; log `logs/slurm/dr_controls-80800.log`.

| Outcome nuisance | Group nuisance | Bias, training 1,250 | Coverage, training 1,250 | Coverage, training 12,500 |
|---|---|---:|---:|---:|
| True probabilities | True probabilities | -0.00005 | 94.78% | 94.88% |
| True probabilities | Linear fit | +0.00009 | 94.80% | 95.30% |
| Linear fit | True probabilities | +0.00012 | 95.40% | 94.58% |
| Quadratic fit | Linear fit | +0.00018 | 93.68% | 95.12% |
| Linear fit | Quadratic fit | -0.00020 | 90.52% | 94.02% |
| Quadratic fit | Quadratic fit | -0.00003 | 94.86% | 94.86% |

Coverage Monte Carlo SE is about 0.3 percentage points near 95% and 0.4
points near 90%. The single-quadratic fits remove the large boosting bias,
supporting the distinction between flexible approximation and a correctly
specified model. They do not automatically restore nominal Wald coverage:
for the linear-outcome/quadratic-group case at equal split sizes, empirical
SD is 0.02411 but average reported SE is 0.02026. Increasing training size
reduces this discrepancy (SD 0.02082, reported SE 0.02018). These results are
consistent with the first-order training contribution identified above.

## Implications for the paper

Retain the double-robust consistency result. Describe the boosting experiment
as a finite-sample robustness comparison, using the actual learner names.
Use the controls to test the theoretical distinction explicitly. Apply the
nominal-coverage claim only under the stated rate and score-convergence
conditions. In particular, an EIF variance is not a universal correction for
all training-sample uncertainty under arbitrary nuisance misspecification.

If coverage with one misspecified nuisance is a separate objective, it needs
additional methodology or assumptions: for example, including training-fit
influence terms for correctly specified parametric fits, or deriving an
appropriate drift-corrected estimator. Simply restoring the old, inflated
variance is not a valid solution. A full refitting bootstrap would also need
its own validity conditions for the chosen learners.
