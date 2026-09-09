# Regeneration after the EIF variance corrections

This report records the reruns requested after the September 9 EIF audit.
Historical outputs and manuscript sections are retained under
`experiments/out/eif_regeneration/before/`; new results and paired checks are
under `experiments/out/eif_regeneration/`. The report distinguishes changes in
interpretation from changes in precision alone.

| Artifact | Change in interpretation |
| --- | --- |
| Figure 1 | No longer supports conservative or >95% coverage claims; pooled containment falls from 247/248 to 229/248. |
| Figure 2 | One boosting nuisance does not give nominal coverage: corrected coverage falls to 52–59% at the largest size. |
| Figure 4 / Experiment 5 | TL needs much more training data; coverage is 79.3% at 250, 94.7% at 2500, and 92% at 10000. |
| TMLE comparison | Strong conservatism disappears; 90–91.5% small-sample coverage exposes undercoverage. |
| Real-data table (manuscript Table 2) | Adult thresholded equal opportunity now excludes zero; other stored metrics retain their interval exclusion status. |
| Exploratory Experiments 1 and 2 | TL is neither uniformly conservative nor uniformly better calibrated than the model-based alternatives. |
| Archived propagation experiment | Greater detection, but changed conditional selection fractions; model-disparity and accuracy curves are unchanged. |
| Figure 3 and new Figures 5–6 | No further interpretation change; these already used corrected probabilistic variance. |

## Scope and additional correction

The original correction covered probabilistic parity and opportunity, including
one-step, NumPy TMLE, and JAX TMLE standard errors. The full dependency audit
found the same group-centering error in **thresholded parity and opportunity**
and their duplicate simulation implementations. Those have now been corrected.
Thresholded opportunity also previously removed observations with Y=0 from the
EIF array while using probabilities estimated over all evaluation observations.
The corrected EIF is evaluated on every evaluation row, with zero contributions
outside the positive-outcome strata. These changes leave the point-estimator
formulas unchanged.

Tests check the thresholded intervals against the variance of a difference in
two group means, including unequal group sizes and constant predictions. The
existing tests check numerical derivatives of the augmented ratio estimator.
The complete local suite passes: 31 tests, with one optional JAX module skipped.
JAX is not installed; its corrected implementation was not executed here.

The regeneration covers the supported manuscript workflow and the saved
standalone experiments. Historical scratch notebooks are not executed: they
retain obsolete APIs and inline formulas, and are explicitly marked archival
in `notebooks/README.md`. They are not sources for current manuscript claims.

CMI, the CMI/TMLE comparison, and conditioning-set experiments do not use these
ratio EIFs and are unaffected. Figures 5 and 6 already use corrected
probabilistic intervals and have not been changed by this regeneration.

## Interpretation changes

### Figure 1: thresholded Setting 1

Point estimates reproduce the archived results to numerical precision. The
mean ratio of corrected to old standard errors is 0.634. Previously 247/248
intervals contained the truth; now 229/248 do (99.6% versus 92.3%). Each point
uses a different sample size, so these are descriptive pooled counts, not
coverage estimates at a given sample size. The manuscript no longer claims
conservative intervals or greater-than-95% coverage across sizes from this
figure. The point-estimate convergence pattern is unchanged.

### Figure 2: Setting 3 robustness

The claim of nominal coverage whenever either nuisance is correctly specified
is not supported by the corrected intervals. A boosting fit is also not an
oracle nuisance function: it retains estimation error. Plot labels now state
which nuisance uses boosting or linear regression explicitly.

A paired rerun calculates both historical and corrected intervals from exactly
the same observations and fitted models. The boosting tie-breaking RNG is now
seeded; the historical script seeded data draws but not this RNG. Full paired
coverage and width results are in `robust_paired.csv`, with replicate-level
results in `robust_paired_replicates.csv`.

With the paired, seeded rerun, coverage for both boosting models is 84%, 90%,
94%, and 94% across total sizes 250, 500, 1000, 2500. With only the outcome
model boosted it is 79%, 69%, 72%, 52%; with only the group model boosted it is
79%, 80%, 72%, 59%. Both linear models have zero coverage at all four sizes.
The historical formula on these exact same fits gives 97–98% and 93–97%
coverage in the one-boosting cases. Thus the failure is attributable to the
variance correction, not Monte Carlo redraws. The earlier unseeded rerun
reported 54% rather than 52% in the largest outcome-boosting configuration;
the manuscript uses the reproducible paired results.

### Figure 3: variance comparison

This figure was already regenerated in the preceding audit. The new rerun
reproduces it. At 1000 observations per split, TL variance is 0.0006292 versus
0.0005679 for the fixed-model baseline: approximately 11% greater variance or
5% greater interval width. The original large variance gap was partly caused
by the EIF bug and a separate baseline denominator bug. No further change in
interpretation relative to the preceding corrected manuscript is needed.

### Figure 4 / Experiment 5: training-size comparison

The old claim that TL reaches nominal coverage around 250 training observations
is false after correction. The first five corrected coverage values are 50.0%,
79.3%, 86.0%, 91.7%, and 94.7% at training sizes 100, 250, 500, 1000, and 2500.
At 5000, coverage is 94.0%, and at 10000 it is 92.0%; improvement is not monotone and nominal coverage does not persist at every large size. The default-booster, tuned-booster, and linear
baseline results reproduce the archived values; TL point estimates also match.
Thus these changes isolate interval correction, not changed data or model fits.
The bias-reduction comparison remains, but the correction does not remove all
bias or guarantee small-sample validity. Tuned model intervals can have
comparable coverage to TL at larger training sizes.

### One-step / TMLE / CV-TMLE comparison

This artifact is generated as `fig6_tmle_coverage.pdf` but is not currently
included by the manuscript source (it is not manuscript Figure 6).
All point estimates, bias, and empirical variances reproduce the archived
results to numerical precision. Coverage changes from 99.5–100% to 90–91.5%
at total sample size 100, and 93–96% at sizes 250–2500. At 1000, mean interval
widths change from 0.2403 to 0.1374 for one-step, 0.2402 to 0.1374 for TMLE,
and 0.1699 to 0.0977 for CV-TMLE. The strong conservatism disappears; small-sample
undercoverage is exposed. CV-TMLE retains shorter intervals and lower empirical
variance, but is not uniformly better calibrated.

### Exploratory Experiment 1

With 500 replicates at each size 250, 500, 1000, 2500, 5000, corrected TL
coverage is 93.0–95.8%; naive fixed-model coverage is 92.2–95.2%. TL is not
uniformly conservative or superior. The historical saved run had 200
replicates at three sizes and a different Monte Carlo truth, so old/new
point-estimate differences here are not attributed to the EIF correction.

### Exploratory Experiment 2

The rerun includes all historical sizes (500, 1500, 4000) as well as the current
CLI defaults (250, 500, 1000, 2500), with 300 replicates per size and 200 bootstrap
resamples. Corrected TL coverage is 91.0%, 92.0%, 93.7%, 95.3%, 94.7%, 93.3%
in increasing size order. Correct-model CLT coverage is 91.0–95.0%; correct-model
bootstrap coverage is 92.7–95.7%. Thus TL is not the uniquely valid method and
does not dominate coverage. Misspecified linear standardization still has
zero parity coverage; misspecified adjusted-effect GLMs still falsely detect a
conditional group effect. The two panels target different estimands.

Historical Experiment 2 used 150 replicates and a different Monte Carlo truth;
the new comparisons use the same truth and draws across methods within each
configuration. Small differences among near-95% rates should not be interpreted
as established rankings (binomial MCSE near 95% is about 1.26 percentage points
with 300 replicates).

## Archived propagation experiment

All 7500 sample-size replicates and 1080 capacity replicates were regenerated.
The fitted-model disparities match the archived raw rows exactly, and TL point
estimates match to below 1e-12. Interval widths and detection/selection rates
change. No manuscript figure currently uses this exploratory experiment.

For the smallest disparity (0.025635), TL detection rates at total sizes
500, 1000, 2500, 5000, 10000 change from 1.6%, 0.8%, 2.4%, 2.4%, 16.4% to
9.2%, 11.6%, 17.6%, 32.6%, 61.4%. Thus the earlier appearance of very weak
power is substantially reduced. At the middle disparity (0.087173), detection
at size 1000 increases from 18.6% to 63.6%.

Conditional fair-looking-model fractions can move in either direction because
the selected set of TL detections changes. At tolerance 0.02 and the smallest
disparity, the size-500 fraction changes from 25% (2/8 detections) to 30.4%
(14/46); the size-1000 fraction changes from 50% (2/4) to 12.1% (7/58).
The old 50% conditional fraction was based on only four detections. These
fractions use a nonzero tolerance on the fitted-model disparity; they are not
false-negative rates of a model-fairness hypothesis test. All tolerance-specific
comparisons are saved in `propagation_fluke_comparison.csv`.

In the capacity sweep at the smallest disparity, the one-round fitted model
still has virtually zero disparity (-0.000124), while TL detection rises from
57.5% to 100%. At 500 rounds it rises from 52.5% to 82.5%. This strengthens
the detection example without changing the model-accuracy/disparity curves.
The experiment has no exact population-null setting, so increased detection
cannot establish false-positive control. Full comparisons are saved in
`propagation_capacity_comparison.csv`.

The original plotting code failed on negative error-bar lengths of order
1e-18 from Wilson-interval rounding at the boundary. The plotting adapter
checks that negative values are no larger than 1e-12 in magnitude before
clamping them to zero; no estimates, confidence limits, or simulation rows
are changed. The restored figure uses the original layout and current shared
Matplotlib style, with descriptive titles replacing unsupported universal
claims such as "Only an accurate rule inherits Psi."

## Real-data intervals and manuscript table

The Adult thresholded equal-opportunity conclusion changes: the corrected
95% interval excludes zero. On identical fitted predictions, the estimate
is 0.122116 and the interval changes from [-0.125310, 0.369542] to
[0.082481, 0.161751]. This corrects both centering and the shortened EIF array.
Adult thresholded parity remains positive: [0.165425, 0.186266] becomes
[0.166167, 0.185524], with the same estimate 0.175846.

Law thresholded parity stays positive, with [0.176035, 0.279482] becoming
[0.203797, 0.251720], estimate 0.227759. Law thresholded opportunity also stays
positive, with [0.050687, 0.193450] becoming [0.099949, 0.144188], estimate
0.122069. The claim that Adult and Law differ in whether thresholded
opportunity is significant is removed. Equal-opportunity intervals are still
narrower in Law than Adult, but probabilistic and thresholded interval lengths
should no longer be described as similar within Law.

Law probabilistic parity has corrected interval [0.171822, 0.211502];
probabilistic opportunity has [0.120327, 0.150285]. Their exclusion of zero is
unchanged. TMLE counterparts remain positive as well.

Adult probabilistic parity remains positive, with corrected interval
[0.170407, 0.190566]; probabilistic opportunity has [0.038788, 0.105331].
The TMLE counterparts remain positive. The only change in interval exclusion
of zero across the eight stored metrics in both datasets is Adult thresholded
opportunity. Full old/new comparisons are saved in `real_data_comparison.csv`;
Adult also has an exact same-fit comparison in `adult_paired_intervals.csv`.
CMI column labels in the comparison mean literal exclusion of zero, not a
validated hypothesis test at the nonregular boundary.

Several manually entered manuscript values already disagreed with the saved
full-data results before this correction (for example Law thresholded parity
was written as 0.19 although the saved estimate was 0.2278). Synchronizing the
table fixes this separate transcription drift; those point-estimate changes
are **not** effects of the EIF fix. CMI estimates and intervals are unaffected
by the variance correction. Their negative one-step values near zero and
nonregular null remain a separate limitation; they cannot establish conditional
independence through a Wald interval.

## Reproduction and provenance

`slurm/eif_regeneration.sbatch` dispatches the main reruns. All jobs use the
whartonstat CPU partition, the existing project virtual environment, and a
single combined `.log` file per job. Paper-scale replicate counts are retained
or increased; raw datasets and nuisance model specifications are unchanged.
The threshold data helper reuses one outcome fit for parity and opportunity
and saves its evaluation predictions, allowing paired old/new interval checks.
The Adult paired helper fits the outcome, group, and joint models once each
and reuses those predictions across six metrics, preserving the SuperLearner
specification and data split. Its threshold results exactly match the separate
threshold rerun.

The obsolete propagation experiment has no surviving source file, but its
Python 3.13 bytecode survives. A copy is archived at
`before/exp6_propagation.cpython-313.pyc`; it calls the current corrected
`tlfair.metrics.prob_parity`. Its full original default grid was rerun
using that artifact rather than inventing a replacement experiment. This is
a reproducibility limitation: restoring its source is still preferable.

## Final validation

All requested supported-workflow and standalone reruns completed. The propagation
job finished all computations but its plot required the rounding-only repair
described above. The artifact validator passes on `data/generated`; all 31
local tests pass (one optional JAX module skipped). The manuscript compiles
successfully, and the regenerated figures and real-data table were visually
checked in the PDF. `simulation_comparison.csv`, `real_data_comparison.csv`,
the paired robustness results, and the propagation comparisons retain the
full numerical audit. `manifest.json` records job IDs and source hashes.
