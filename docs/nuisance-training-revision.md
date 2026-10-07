# Matched nuisance-training revision

The prespecified revision uses independent standard-normal shared and outcome-only
predictors, plus three noise predictors. The normalized linear/quadratic transform
has lambda 0.5 in the main experiment and 0 and 0.75 in sensitivity checks. Group
association and outcome signal are both 1; shared outcome signal is 0.6. The
original five-feature, squared-only process remains a separate stress test.
These parameters are fixed before reviewing comparative coverage.

The main grid has training sizes 100, 250, 500, 1000, 2500, 5000, 10000. Controls
and sensitivity settings use 500, 2500, 10000. Every configuration has 1000
independent replicates and evaluation size 2000. Default boosting uses 100 trees,
depth 3, learning rate 0.1. Higher-capacity boosting uses 1000 trees, depth 5,
learning rate 0.05; this is a fixed configuration, not cross-validated tuning.

On 2026-10-06, the CV comparison was deferred at the user's request. The active
figure, appendix tables, and default workflow again use only the fixed learner
results in `data/generated/nuisance_training/`. CV code, checkpoints, and collected
outputs remain available; the three-strategy manuscript is archived in
`results/data/nuisance_revision_with_cv/`. The historical CV record below is retained.

## Early-stopping revision (2026-10-06)

The user subsequently requested early stopping for both boosters. Both learning
rates remain 0.1; depth/maximum-tree-count pairs are (3, 100) and (5, 1000).
Both use the built-in scikit-learn stopping procedure with a stratified 20%
training holdout, `n_iter_no_change=20`, and `tol=1e-4`. Outcome and group fits
stop independently; identical role-specific seeds give the two configurations
the same internal split. The final fit is the built-in stopped ensemble trained
on the other 80%, without full-training refitting or best-iteration restoration.
The 2000 evaluation observations are never used for stopping. A small training
sample also means a noisy validation sample; nominal coverage is not guaranteed.

`experiments/nuisance_early_stopping.py` reruns all 16 settings on their original
draws, including every mixed boosting/linear comparison. Linear/quadratic/oracle
fits are unchanged and checked against the original prediction hashes. Both
boosting strategies must be refit, so the previous default results are not reused.
Replicates retain actual tree counts for each nuisance, evaluation probability
MSE against known truth, and evaluation Brier scores, calculated after fitting.
Settings are fixed before examining the new fairness results. The original
fixed-tree code and all completed checkpoints remain available separately.

Cluster job 89083 runs the early-stopping revision from
`/shared_data0/jrudoler/tlfair_early_20261006` in 320 shards of 50 replicates.
Superseded job 88979 was cancelled after the new job was submitted; its
completed checkpoints were retained. All 51 tests passed.

The current workflow targets `data/generated/nuisance_early_stopping/`. Run
`analysis/sim_nuisance_early_stopping/run.py --n-jobs 5` locally, or submit
`slurm/nuisance_early_stopping.sbatch` from an isolated cluster snapshot. Collect
with `experiments/nuisance_early_stopping.py collect --input-dir
 data/generated/nuisance_early_stopping_shards`, then run the default validation,
figure, and table scripts. The completed run was collected and validated on 2026-10-07; Figure 2, appendix
tables, and the compiled manuscript now use these results.

## Matched learning-rate revision (2026-10-06)

The user requested learning rate 0.1 for both fixed boosters. The default remains
100 depth-3 trees; the higher-capacity configuration remains 1000 depth-5 trees.
`experiments/nuisance_matched_rate.py` reruns only the higher-capacity fits on all
16 original settings, reusing the original draws, seeds, default fits, controls,
and population integrations. The original source and outputs remain frozen.
The change is a post hoc design revision, not a newly prespecified experiment.

Cluster job 88979 runs 640 independently resumable shards of 25 replicates in
`/shared_data0/jrudoler/tlfair_rate_20261006`, with five workers per shard and at
most 24 concurrent jobs. New results go to
`data/generated/nuisance_matched_rate/`; validation checks unchanged retained
rows, matched observations, source/runtime consistency, and paired rate contrasts.
The pre-revision manuscript is archived in
`results/data/nuisance_revision_before_matched_rate/`.

Reproduce the replacement locally with
`analysis/sim_nuisance_matched_rate/run.py --n-jobs 5`, or submit
`slurm/nuisance_matched_rate.sbatch` from the isolated cluster snapshot. Collect
completed checkpoints with `experiments/nuisance_matched_rate.py collect`.
The default figure/table workflow now targets these replacement outputs; the
existing PDF still describes the old results until the rerun is collected.
All 49 tests passed before submission.

## Original fixed-grid reproduction

Run the fixed-learner grid locally (expensive):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python analysis/sim_trainsize/run.py --n-jobs 8
```

The fixed-learner Slurm entrypoint is `slurm/nuisance_training.sbatch`, configured for Locust's
standby partition. Submit from an isolated snapshot directory on shared storage,
with `logs/` already present. Set `TLFAIR_PYTHON` to the desired compatible
scientific Python executable. The array uses 320 shards of 50 independently
seeded replicates, with at most 16 concurrent eight-CPU jobs. Seeds depend on
scenario, sample sizes, and replicate ID, not job order or worker count.
The CV extension uses `slurm/nuisance_tuning.sbatch`: 280 shards of 25 replicates,
five workers each, with at most 24 concurrent jobs.

Completed replicates are atomically checkpointed. Rerunning a shard skips them;
metadata prevents resuming with different source hashes or runtime versions.
Collect and validate after the fixed-learner shards finish:

```sh
.venv/bin/python experiments/nuisance_training.py collect \
  --input-dir data/generated/nuisance_training_shards \
  --output-dir data/generated/nuisance_training
.venv/bin/python analysis/nuisance_tables/validate.py data/generated/nuisance_training
# To render the current matched-rate revision after collecting it:
.venv/bin/python analysis/nuisance_tables/validate.py
.venv/bin/python analysis/fig_trainsize/run.py
.venv/bin/python analysis/nuisance_tables/run.py
```

The standard Snakemake targets use the same simulation, figure, and table scripts.
Generated tables are copied into `paper/tables/` by `sync_nuisance_tables`.
The old heatmap script is retained for historical reproduction but is no longer
part of the manuscript's default figure dependencies.

## Statistical reporting

Signed bias is the mean estimate minus the population data-fairness target.
Bias, estimator SD, and reported SE are shown in percentage points. Coverage and
bias Monte Carlo standard errors use independent replicates; paired differences
retain covariance between TL and model-fairness results. The plot shows
plus/minus 1.96 MCSE, with coverage bars bounded to [0,1]. At empirical coverage
zero or one, the plug-in MCSE is zero; this is not proof of an exact probability.
These error bars describe simulation precision, not intervals within a replicate.

Eight independently scrambled Sobol integrations provide truth and population
characteristics. Integration resolution is increased until its estimated SE is
below 1% of the smallest bias MCSE for that scenario. The full CSV retains both
5th/95th and 1st/99th propensity quantiles and group/outcome prevalence.

For each replicate, outcome predictions are fit once and shared by both inference
methods. Data and prediction hashes provide an audit of this matching. Oracle
probabilities have no fitting error. Quadratic logistic fits include raw and
squared features and an intercept, with C=1e8 and convergence checked; they are
correctly specified but still estimated.

## Provenance

Previous manuscript sections, figures, scripts, and recovered plotting values
were preserved before this revision in `results/data/nuisance_revision_before/`,
with SHA-256 hashes. They are historical artifacts, not input to the new plots.
The original experiment remains in `experiments/exp5_trainsize_coverage.py` for
historical reproduction. New computation is in `experiments/nuisance_training.py`.
Run metadata and source hashes are retained with the new raw results and manifest.

The October 5 run uses an isolated source snapshot at
`/shared_data0/jrudoler/tlfair_nuisance_20261005` on Locust. Verification job
88031 and array 88044 completed all 16,000 independent datasets (118,000
method/configuration rows). Collection job 88068 crashed in the cluster runtime
before writing outputs; all checkpoints were transferred intact and collection
and validation completed locally. The simulation environment uses Python
3.12.11 and scikit-learn 1.8.0; each shard records its numerical-library versions
and source hashes. Initial array 87697 failed before computation because its
staging directory was node-local; it produced no results.

Local collection used Python 3.13.9, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3,
and scikit-learn 1.9.0. It did not refit any learners. The eight-scramble Sobol
integration used power 16 for each continuous-family setting and power 23 for
the squared-feature stress test, meeting the prespecified precision criterion.
The full validation rebuilt all 118 summaries and paired comparisons from
saved replicates and checked shared observations and predictions. All 44 tests
passed, including worker-count reproducibility. Completed checkpoints are also
available locally in `data/generated/nuisance_training_shards/`; consolidated
replicates and the manifest are in `data/generated/nuisance_training/`.

## Post-hoc bias mechanism check

`analysis/nuisance_tables/diagnose_bias.py` refits the first 16 primary-setting
seeds at training size 500 using the local runtime recorded above and evaluates
on 16,384 scrambled-Sobol fresh predictors. Its output is
`results/data/nuisance_bias_diagnostic.csv`; it is not an input to the figure,
tables, or prespecified simulation results. Higher-capacity fits nearly
interpolated the training labels but had roughly four times the fresh-predictor
probability MSE of default boosting for both nuisances. Positive alignment of
outcome and group probability errors yielded a negative TL population remainder.
The diagnostic reports the model-fairness error projection and TL product-error
remainder separately; the latter omits finite-evaluation ratio bias. With only
16 fits, these are mechanism checks, not replacements for the reported 1000-replicate
bias estimates. No simulation settings were changed after these checks.

## Cross-validated boosting extension

Following inspection of the two fixed configurations, the user requested a third,
genuinely tuned learner. The previous experiment's label “tuned” described the
same fixed 1000-tree/depth-5/learning-rate-0.05 configuration now called
higher-capacity; it did not use a tuning search. The new extension keeps the
population, all seven primary training sizes, evaluation size, 1000 replicates,
and random draws unchanged. Existing fixed-configuration results are reused.
The prior two-configuration manuscript and figure are archived in
`results/data/nuisance_revision_before_cv/`.

Three stratified folds select average held-out log loss separately for the outcome
and group models. The fixed ten-candidate search includes depths 1, 3, and 5 crossed
with 100, 500, and 1000 trees, at learning rate 0.05, plus the default booster
(100 trees, depth 3, learning rate 0.1). Both fixed comparators are candidates.
Staged predictions reuse each fold's tree sequence across tree counts. The selected
configuration is refit on the whole training sample. No evaluation outcomes or
fairness results select hyperparameters. Selection does not guarantee improved
population accuracy, particularly with small training datasets.

The extension script is `experiments/nuisance_tuning.py`. The local wrapper is
`analysis/sim_nuisance_tuning/run.py`; `slurm/nuisance_tuning.sbatch` runs 280
independently resumable 25-replicate shards. All fold losses, selected parameters,
model seeds, and CV seeds are checkpointed. New outputs are collected separately
in `data/generated/nuisance_training_cv/`, retaining the original full experiment
and adding 14,000 tuned-method rows. Only the primary setting receives the CV
extension; the previously specified sensitivity and misspecification experiments
remain the fixed-learner controls.

```sh
.venv/bin/python analysis/sim_nuisance_tuning/run.py --n-jobs 8
# Or collect completed cluster shards without fitting again:
.venv/bin/python experiments/nuisance_tuning.py collect \
  --input-dir data/generated/nuisance_tuning_shards
.venv/bin/python analysis/nuisance_tables/validate.py data/generated/nuisance_training_cv
.venv/bin/python analysis/fig_trainsize/run.py
.venv/bin/python analysis/nuisance_tables/run.py
```

The CV run uses the isolated Locust snapshot
`/shared_data0/jrudoler/tlfair_cv_20261005`, with initial shard job 88405 and
remaining-array job 88415. It uses the same Python/scientific-library versions
and original simulation source hashes as the fixed comparisons. Resume fitting
in that environment; collection and rendering can run locally without refitting.

Pending CV shards were subsequently resubmitted as array 88471 with five workers
per 25-replicate shard (at most 24 simultaneous jobs). This avoids a mostly idle
last worker batch and fits smaller free allocations. Running shards were retained;
seeds and tuning settings did not change. Slurm automatically requeues preempted
standby jobs, and the shard runner skips completed records. A redundant retry job
88432 was cancelled immediately once the automatic retry was observed.

The completed CV checkpoints were retrieved on 2026-10-06: 7,000 independent
primary-setting replicates, 14,000 method rows, and 14,000 nuisance selections.
The combined delivery contains 132,000 method rows and 132 summaries. Full-run
validation regenerated every summary and paired comparison, verified common
observations and outcome predictions, checked source/runtime consistency, and
confirmed that selections matching a fixed comparator reproduce its predictions.
The full test suite passed (47 tests), including worker-count reproducibility
and exclusion of evaluation observations from hyperparameter selection.

CV selected depth-1 trees in 976/1000 outcome fits and 998/1000 group fits at
training size 1000. At that size, model-fairness/TL coverage was 0.318/0.920 and
signed bias was -2.21/-0.44 percentage points. Predictive tuning therefore did
not uniformly improve inference on the population fairness target. The manuscript
reports this outcome alongside both fixed configurations without claiming that
capacity or predictive selection guarantees better fairness inference.

The final Figure 2 and appendix tables were regenerated from the collected
replicates. The manuscript compiled with XeLaTeX without undefined references or
overfull boxes, and the affected pages were visually inspected. Final hashes and
collection metadata are in `data/generated/nuisance_training_cv/delivery_manifest.json`.
The original fixed-configuration outputs and pre-CV manuscript archive remain
available separately.


## Early-stopping completion and numerical control audit (2026-10-07)

All 320 jobs completed: 16,000 datasets, 118,000 method rows, and 118 summaries.
All summaries and paired Monte Carlo comparisons were regenerated from saved
replicates; population integration uncertainty remains negligible. The data hashes
match the original experiment in every configuration. All 52 tests pass, and the
manuscript compiled without undefined references or overfull boxes. The figure
and affected manuscript/appendix pages were visually inspected.

Refitted linear and quadratic models on some cluster shards did not reproduce
bitwise-identical prediction hashes, despite matching seeds, data, source, and
library versions. The largest unchanged-control point-estimate difference was
2.85e-9; across estimates, endpoints, widths, and standard errors the maximum was
3.44e-9. Coverage was identical for all 45,000 comparable method rows. Oracle
hashes matched exactly. The collector now validates these numerical quantities
with absolute tolerance 1e-8 instead of requiring identical fitted-logistic hashes;
it still rejects changed observations, oracle predictions, coverage decisions,
or material numerical differences. `control_audit.json` records the check.
The computation source is preserved in shard metadata and
`results/data/nuisance_revision_before_early_stopping/simulation_source.py`;
only collection validation was repaired after the runs. This is consistent with
small floating-point differences between refits; the specific hardware cause
was not established. Resume any original shards using their frozen cluster source.

`analysis/nuisance_tables/stopping_diagnostics.py` regenerates
`stopping_summary.csv` from the saved replicates, including tree counts, fraction
at the tree limit, probability MSE, and evaluation Brier scores. At training size
1000, median outcome/group tree counts are 38/34 (lower capacity) and 29/27
(higher capacity). Higher-capacity mean probability MSE remains larger for both
nuisances. Model-fairness bias is negative for both configurations in the primary
setting, whereas TL bias is much smaller. These revised fits change learning rate
for higher capacity and introduce a validation split and early stopping for both;
comparisons with the old figure cannot isolate the effect of early stopping alone.

The pre-revision manuscript is preserved in
`results/data/nuisance_revision_before_early_stopping/`. Final artifact hashes are
recorded in `data/generated/nuisance_early_stopping/delivery_manifest.json`.
