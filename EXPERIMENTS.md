# Manuscript analyses and reproduction

Use semantic names; LaTeX supplies figure and table numbers. The publication
workflow renders frozen, checksummed results. New computation is a separate,
explicit workflow with a fresh run ID.

The sample-variance regeneration requested on 2026-10-23 uses run ID
`variance-nminus1-20261023`. All nine analyses are rerun from saved source
snapshots with `ddof=1` for the evaluation influence-function variance. The
conditioning-set figure displays the mean endpoints of two-sided 95% CMI
intervals; its rejection rates still use the one-sided rule
`estimate - 1.645 * se > 0`. The bands summarize replicate intervals, not
uncertainty in the Monte Carlo mean.

[config/variance-publication.json](config/variance-publication.json) records
the approved sources for the selected publication
`manuscript-2026-10-23-sample-variance`. All nine analyses and the manuscript
build completed on 2026-10-26; the current-code provenance audit verifies all
nine selected runs. Seven of the original analysis jobs
100876–100884 completed; the nuisance reference hit the home-directory quota
while collecting its 16,000 saved replicates. After space was freed, recovery
job 105571 resumed the original saved source, checking that every existing
replicate remained unchanged. Early-stopping job 105572, comparison job 105573,
and guarded completion job 105574 then completed. The completion job
validated the CMI records, selected all nine certified runs, validated the full
publication, and rebuilt the manuscript. It refuses changed source files or
an unexpected publication selection and can resume after a failed build.
The submission receipt is
`results/provenance/submission-variance-nminus1-20261023.json`; the recovery
submission is recorded separately in
`results/provenance/recovery-submission-variance-nminus1-20261023-20261026.json`.
The successful build and PDF checksum are recorded in
`results/provenance/completion-manuscript-2026-10-23-sample-variance.json`.
The selected inputs and their checksums are tracked in `config/publication.json`
and `config/publication-provenance/`; previous inputs remain available locally
under `data/imported/publication-before-manuscript-2026-10-23-sample-variance/`.

All selected CMI analyses use Figure 2's depth-3 boosting protocol: at most
100 rounds, learning rate .1, a stratified 20% validation subset within training,
patience 20, and tolerance 1e-4. The joint four-class classifier uses no probability
calibration or full-training refit. `tlfair/cmi_estimation.py` defines the shared
learner; simulation replicate records retain the model seed, stopped round count,
and tree count, and real-data fit metadata records the full configuration.
The CMI protocol reruns use run ID `cmi-depth3-20261021` and publication ID
`manuscript-2026-10-21-cmi-depth3`. Earlier runs and diagnostic source snapshots
retain the older fitting procedures for historical reproduction.
The four completed production runs execute the saved source from commit `05ea0c5`:
Slurm jobs 98244 (Adult), 98245 (Law), 98246 (CMI grid), and 98247
(conditioning sets). Simulation grids, replicate counts, random streams,
dataset versions, and outer train/evaluation splits are unchanged.
All 11,312 TMLE fits converged. Validation reconstructs the simulation summaries
and real-data intervals from their saved records. All 300 datasets shared with
the earlier Figure 2 protocol review reproduce its estimates, intervals, and
stopping rounds; simulation truths and KNN estimates match the October 19 runs.
Comparison and validation reports are under `results/review/cmi-depth3-20261021/`.
Only the four CMI run selections changed; the other five analyses retain their
verified inputs. The selection receipt is
`results/provenance/selection-manuscript-2026-10-21-cmi-depth3.json`.

## Current manuscript artifacts

| Analysis ID | Rendering target | Manuscript artifact |
|---|---|---|
| `parity_retraining` | `fig_parity_retraining` | `figs/parity_retraining.pdf` |
| `nuisance_early_stopping` | `fig_parity_nuisance_training` | `figs/parity_nuisance_training.pdf` |
| `cmi_conditioning_set` | `fig_cmi_conditioning_set` | `figs/cmi_conditioning_set.pdf` |
| `cmi_estimation` | `fig_cmi_estimation` | `figs/cmi_estimation_{error,coverage}.pdf` |
| `real_data_adult`, `real_data_law` | `table_real_data_inference` | `tables/real_data_inference.tex` |
| `cmi_estimation` | `table_cmi_population_truth` | `tables/cmi_population_truth.tex` |
| CMI paired methods | `cmi_paired_comparison` | `figs/cmi_paired_comparison.pdf`, `tables/cmi_null_diagnostics.tex` |
| `nuisance_early_stopping` | `nuisance_tables` | `tables/nuisance_specification.tex` |

Paths in the last column are relative to `paper/`. Results are rendered under
`results/figures/` and `results/data/`, then copied with the same basename.
The real-data and CMI-truth tables are generated inputs; captions stay in the
manuscript. Appendix G includes one specification table, with TL coverage and
signed bias at the smallest and largest training sizes for all oracle/quadratic
controls and the doubly misspecified linear baseline. The capacity comparison
remains in the main figure. Full population, fitted-learner, control, and
sensitivity tables, `nuisance_findings.tex`, and `nuisance_tables.csv` remain
reporting aids under `results/data/`; they are not delivered to the manuscript.
`stopping_summary.csv` retains the learner-fitting diagnostics. The
`nuisance_reference` analysis supplies the fixed-tree control audit
used by early-stopping collection and validation; it is an active dependency.

## Publication inputs and provenance

- [config/publication.json](config/publication.json) identifies each approved
  input by path, SHA-256, and completed-run record. It includes canonical inputs
  under `data/generated/` and `data/raw/`, native CMI replicate records, and
  small tracked completion/manifests under `config/publication-provenance/`.
  These inputs are not overwritten by new computation.
- [config/artifacts.json](config/artifacts.json) maps every included figure/table
  to its data, rendering code, shared modules, style, lockfile, and manuscript path.
- [config/locust-recovery.json](config/locust-recovery.json) inventories the 96
  files recovered from Locust on 2026-10-09, including historical source/config
  snapshots. The 1.36 GB bundle is `data/imported/locust-2026-10-09/` (not in Git).
  The recorded source commit is the checkout observed at recovery, not proof of
  the commit originally used to compute every historical file.
- `paper/artifact-manifest.json` records delivered output and dependency hashes,
  rendering runtime, and Git state. It is versioned with the manuscript.
- [config/datasets.json](config/datasets.json) pins external raw data URLs to Git
  revisions and records byte checksums. Download helpers reject conflicting files.

Before the October 19 TMLE transition, the selected publication was
`manuscript-2026-10-14`, selected from seven
completed analyses in `data/runs/reproduction-20261012/` and two matched real-data
analyses in `data/runs/matched-real-data-20261014/`. Each ran its saved
source snapshot with the current computation dependencies and recorded package
versions. Exact source hashes certify computations made before the workflow
changes were committed; the recorded Git commit alone is not used as proof.

Validation checks 118,000 nuisance rows and their matched reference/control
audits, reconstructs the CMI estimation and conditioning-set summaries from
saved replicates, and can reconstruct all 405 retraining summary rows. The
Adult and Law inputs are fresh, full-precision runs of every manuscript metric.
The CMI inference limitations remain statistical qualifications, not gaps in
the selected computations' provenance.

The approved data/model comparison uses `analysis/matched_real_data/run.py` to
fit one outcome model per dataset and reuse its held-out predictions for both
targets. Probabilistic parity and equal opportunity precede the thresholded
metrics in the table. Native `predictions.npz` and `fit_metadata.json` records
allow both sets of intervals to be reconstructed without refitting. CMI uses
the separately selected joint-classifier run and has no model-fairness column entry.

Both matched fits completed on October 15: Slurm jobs 94570 (Adult) and 94571
(Law). Dependent job 94588 selected the verified results, reconstructed the
intervals from cached predictions, and delivered the table and approved prose.
Its PDF compilation failed because the compute node lacked Perl `Time::HiRes`.
On October 19, `snakemake verify_computation paper --cores 1 --quiet` completed
successfully on the login environment; the resulting table was visually checked.
That publication ID is `manuscript-2026-10-14`. Completion is recorded in
`results/provenance/selection-manuscript-2026-10-14.json`; the recovered build log
is `logs/matched-paper-check-20261019.log`. The prior selection is retained in
`data/imported/publication-before-manuscript-2026-10-14/`.

The guarded completion script is for the original one-time selection. After
selection has succeeded, recover a failed render/compile using the ordinary
Snakemake build, rather than rerunning the selection script.

### CMI TMLE transition

The approved October 19 change makes TMLE the primary CMI estimator throughout
Adult/Law, the coupling simulation and the conditioning-set simulation. The
one-step derivation remains in the manuscript and paired one-step results remain
in the appendix. Parity/EO retain their existing matched, certified results.

Runs are under `data/runs/cmi-tmle-20261019/`: Adult job 97231, Law 97232,
estimation 97233, conditioning set 97234. Adult/Law completed and their estimates
and targeted probabilities reconstructed from saved caches. The two simulations
were submitted with the original grids, data streams, seeds and learners.
They fit the initial joint classifier once per replicate and save paired
`one_step`, `substitution` and `tmle` records. Substitution is a computational
diagnostic; the paper compares one-step and TMLE, with KNN retained as a baseline.
Only TMLE and one-step intervals are interpreted in the paired manuscript outputs.

`config/cmi-tmle-publication.json` records the guarded manuscript/workflow hashes.
Dependent job 97236 runs `analysis/tools/complete_cmi_tmle.py` after the simulation
jobs succeed. It checks those hashes, validates every saved summary and targeting
score, validates both real-data prediction caches, and selects only the four CMI
runs as `manuscript-2026-10-19-tmle`. All other native selected runs are retained.
The updated `adult_results.pkl` and `law_results.pkl` contain CMI-only comparisons;
all four parity/EO rows come from the matched-model comparison pickles.

The actual current selection is always `config/publication.json`. Full completion
is certified by `results/provenance/completion-manuscript-2026-10-19-tmle.json`;
the job log is `logs/slurm/paper-cmi-tmle-97236.log`. Until the new runs are selected,
the strict build rejects older CMI results that do not match the updated code.
The previous compiled paper and all old native runs remain available. A watermarked
syntax/layout check is under `results/review/cmi-tmle-preflight/`; its simulation
figures are previews, not selected manuscript results. None of this review bundle
is tracked in Git.

The new completion script can resume after successful selection if a later build
fails. It refuses to overwrite a different publication or proceed after guarded
source changes. The compiler falls back to direct XeLaTeX/BibTeX passes when the
compute node lacks Perl `Time::HiRes`, which prevented the earlier matched build.

Targeting convergence requires an absolute empirical score at most `1e-8`, with
at most 200 updates and likelihood-increasing line search. Convergence failures
remain in native records and make the run fail. Joint probabilities are floored
and renormalized once, then all marginals/log ratios use that same joint law.
The original one-step log ratio remains unclipped. Symmetric Wald intervals are
nominal; convergence/nonnegativity do not validate their calibration at conditional
independence, where the population EIF and its first-order variance are zero.

The former publication registry and generated inputs are retained outside Git
under `data/imported/publication-before-manuscript-2026-10-13/`. The original
Locust recovery bundle is also retained as historical evidence. Legacy CMI
aggregate/rounding limitations describe that previous selection.

To restore missing canonical copies when the selected native run directories
and pinned raw data are already available (conflicting files cause an error):

```sh
.venv/bin/python analysis/tools/restore_publication.py
```

Retain the native run directories, their source snapshots, and pinned raw data
outside Git. A fresh Git clone contains the code and tracked provenance metadata,
but does not contain the computation outputs or raw data.

## Render, validate, deliver, compile

From the repository root, install dependencies with `uv sync`, then:

```sh
uv run snakemake validate --cores 1
uv run snakemake all --cores 1 --dry-run
uv run snakemake all --cores 1
uv run snakemake deliver --cores 1
uv run snakemake paper --cores 1
# Optional full reconstruction of retraining summaries from raw replicates:
uv run python analysis/tools/validate_publication.py --deep \
  --output results/provenance/deep-validation.json
```

Individual rendering targets in the table above are also available. `all`
validates and renders; `deliver` additionally copies assets and writes their
manifest; `paper` additionally compiles with XeLaTeX via `latexmk`. None launches
simulations or downloads. Missing or changed publication inputs fail explicitly.
The tracked `clean-figs.mplstyle` is the default on every host. To compile only
already-delivered manuscript assets, use
`uv run python analysis/tools/compile_paper.py`.

## New computation

[config/analyses.json](config/analyses.json) is the source of scientific parameters
for new runs. The runner records the resolved configuration, additional learner
and design settings, source snapshot and hashes, Git/runtime/lockfile information,
input and output hashes, command, timestamps, and completion status. Results go to
`data/runs/<run_id>/<analysis_id>/`; an existing directory is never overwritten.
A failed run keeps its log and manifest and needs a fresh run ID to restart.

```sh
# Inspect a new run; put targets before --config.
uv run snakemake --snakefile workflow/compute.smk --cores 8 --dry-run \
  --config analysis=cmi_estimation run_id=cmi-check-20261009 njobs=8 compare_mem_gb=16
# Run only after reviewing the plan (omit --dry-run).
# Equivalent direct invocation:
uv run python analysis/tools/run_analysis.py --analysis cmi_estimation \
  --run-id cmi-check-20261009 --n-jobs 8 --compare-mem-gb 16
```

Do not run both commands with execution enabled for the same ID. Worker count
and memory are resources, not scientific parameters; CMI task streams are
independent of serial/parallel execution and grid ordering. The RNG change does
not rewrite historical publication data. New early-stopping runs require
`--reference-dir data/runs/<reference_run_id>/nuisance_reference` (or Snakemake
`reference_dir=...`) for a matching, completed reference run. Validate the
scientific compatibility of the reference before collecting the audit.

A new run does not automatically become a publication input. Review its results,
validate its summaries, then deliberately update the publication registry and
canonical input copies together. Retain the prior run and manifest.

## Which computation produced the manuscript results?

`config/publication-runs.json` maps each analysis's manuscript input copies to
outputs in a selected completed run directory, with exact completion/manifests
copied into `config/publication-provenance/`. All current selections are certified;
a `null` run or changed computation dependency causes publication validation and
delivery to fail. Recovery checksums alone do not certify computational origin.

New runs execute the source snapshot saved under their own `source/` directory,
not the live checkout. The run manifest records the exact command, scientific
settings, declared computation dependencies (`code` in `config/analyses.json`),
raw input hashes, runtime, Git state, and output hashes. The runner refuses to
complete if the saved source, raw inputs, or nuisance reference change during
computation. A completion record certifies the final manifest's checksum.

From the repository root:

```sh
uv run snakemake audit_computation --cores 1
# Fail unless every manuscript computation is linked to the current code:
uv run snakemake verify_computation --cores 1
# Compare completed new runs with manuscript inputs without selecting them:
uv run python analysis/tools/compare_reproduction.py --run-id reproduction-20261012
```

The audit checks source snapshots, current code and scientific settings, raw
inputs, runtime versions, completed outputs, selected nuisance references, and
the equality of manuscript input copies to those outputs. Every delivered
artifact manifest also includes the computation audit, separately from its
rendering dependencies. An unverified result stays labeled unverified.

After reviewing a new run, select all analyses from that run ID together:

```sh
uv run python analysis/tools/select_publication_runs.py \
  --run-id NEW_RUN_ID --publication-id NEW_PUBLICATION_ID
uv run snakemake verify_computation paper --cores 1
```

Selection verifies completed runs before replacing anything, preserves prior
generated inputs outside Git, and updates the copies, checksums, run links, and
tracked completion/manifests together. Large raw and generated files stay
outside Git; the code, configurations, and lineage metadata are tracked.

Comparison tables report differences rather than interpreting a checksum change
as a scientific change. CMI null diagnostics separately report two-sided coverage,
positive and negative exclusion, and the diagnostic one-sided positive rejection
rule. Normal critical values are not asserted to be calibrated at independence.

## Locust

The synchronized Locust checkout is `/home/jrudoler/tl_fairness`. Its publication
inputs, Python environment, figure rendering, and both workflow dry runs have
been checked. A fresh worktree needs its own environment and access to the
untracked data. Manuscript compilation on Locust was verified on 2026-10-12
after updating TinyTeX's package manager (`tlmgr update --self`) and installing
the missing math fonts (`tlmgr install libertinus-fonts`). The existing XeLaTeX
build produces `paper/main.pdf`. After fixing a previously failed build,
`latexmk -g -xelatex -interaction=nonstopmode -halt-on-error main.tex` from
`paper/` forces a fresh attempt if latexmk has cached the earlier failure.
The recovery bundle's repository path is a symlink on Locust to
`/shared_data0/jrudoler/tlfair_publication_recovery_20261009`; all 96 files were
checksum-verified during relocation after the shared `/home` filesystem filled.
Check storage availability before launching new runs, which otherwise write
under the checkout's `data/runs/` directory.

Use Locust for expensive new computation:

```sh
mkdir -p logs
sbatch slurm/analysis.sbatch cmi_estimation cmi-check-20261009
sbatch slurm/parity_retraining.sbatch parity-check-20261009
sbatch slurm/analyze_fulldata.sbatch real-data-check-20261009
```

Adjust allocation resources/time to the analysis before submitting; the scripts'
allocations are defaults, not runtime guarantees. `slurm/figures.sbatch` renders
existing publication data only. No cluster job is submitted by publication builds.
The nuisance array/collection launchers are retained for historical shard runs;
use them only with their matching frozen source snapshots. Shared-data snapshots
from October 5–6 are preserved separately from the working checkout. For new
configured runs, use the generic launcher and the new-run manifest workflow.

## CMI learner diagnostic comparison (2026-10-21)

`analysis/cmi_learner_review/run.py` compares nuisance fitting choices without
altering the publication selection or manuscript. The fixed plan replays all
100 native coverage datasets at n=10,000 for kappa=0, 2.5, and 4. The original
calibrated boosting estimates and intervals must match the October 19 run.

Comparisons include the October 19 three-fold sigmoid-calibrated ensemble, the
same three base fits without calibration, a full-training default fit with and
without sigmoid calibration, tuned boosting with and without calibration,
smooth multinomial logistic regression, and the analytic oracle joint law.
All CMI estimators share each learner's joint predictions and evaluation data.

The tuned learners use a stratified 20% holdout inside the training half of the
sample. Boosting considers depths 1, 3, and 5, minimum leaf size 20, learning
rate .05, plus the original depth-3/leaf-size-1/rate-.1 path. Each path checks
validation log loss every 25 rounds through at most 1,000 rounds and stops
after four checks without improvement of at least 1e-5. The best observed
configuration is refitted on the full training half. The smooth learner selects
polynomial degree 1, 2, or 3 and C=.1, 1, or 10 by the same holdout loss, then
refits on all training rows. Polynomial features include interactions; no true
simulation index or probabilities are supplied to these learners. Calibrated
full fits use three-fold out-of-fold scores and `ensemble=False`. Their
hyperparameters have been selected within training before calibration CV.

```sh
XDG_CACHE_HOME=logs/cache .venv/bin/snakemake \
  -s workflow/cmi_learner_review.smk --cores 4
```

For cluster execution, first run the `prepare` command, then submit
`sbatch slurm/cmi_learner_review.sbatch`. After the array completes, run
`MPLCONFIGDIR=logs/matplotlib .venv/bin/python analysis/cmi_learner_review/run.py collect`.
The launcher uses whartonstat, at most four concurrent four-CPU tasks, and
merged logs under `logs/slurm/`. Paths are configurable in the Python CLI and
the diagnostic Snakemake workflow. A different plan or source requires a fresh
run directory.

Source snapshots, runtime and input hashes, per-replicate tuning audits, data
hashes, and checksummed completion records live outside Git under
`data/runs/cmi-learner-review-20261021/`. Reports and figures are under
`results/review/cmi-learner-review-20261021/`. Accuracy compares predictions
with the analytic probabilities at evaluation covariates; targeted accuracy
uses the evaluation labels and is not an independent validation score. Exact
CMI truth comes from one-dimensional quadrature; coverage against the original
Monte Carlo truth is also retained. Oracle Wald inference at kappa=0 is
degenerate and cannot be interpreted as a regular nominal-coverage control.

The exact Figure 2 protocol is a separate extension:

```sh
XDG_CACHE_HOME=logs/cache .venv/bin/snakemake \
  -s workflow/cmi_figure2_review.smk --cores 8
```

`slurm/cmi_figure2_review.sbatch` launches this extension on whartonstat.
It imports the two configurations directly from
`experiments/nuisance_early_stopping.py`: depth 3 with at most 100 rounds,
or depth 5 with at most 1,000 rounds, both learning rate .1, a stratified
20% validation split, patience 20 and tolerance 1e-4. There is no probability
calibration, configuration selection, or full-training refit. The models fit
the four-class joint response, so each round adds four trees. Dataset hashes
must match all 300 datasets in the earlier learner review. Sources and
per-replicate outputs are in `data/runs/cmi-figure2-review-20261021/`;
reports are in `results/review/cmi-figure2-review-20261021/`.

## Retired work and naming history

[The archive index](archive/workflow_2026-10-09/README.md) records inactive analyses,
including older parity/misspecification/TMLE figures, permutation importance,
nuisance CV and matched-rate extensions. They are absent from the publication
workflow. The historical separate-fit naive comparison is superseded by the
approved shared-prediction comparison above. Historical code and reports remain available without becoming build
dependencies.

[The naming archive](archive/naming_2026-10-09/README.md) preserves changed source
snapshots, old-to-new names, and figure rename checksums. In particular,
`fig9_retraining`, `fig7_trainsize`, `fig8_condset`, and `fig4_cmi` were replaced by
the semantic rendering targets above. Historical reports retain their original
names. The original audit is in [docs/workflow-audit.md](docs/workflow-audit.md);
its follow-up distinguishes resolved issues from historical provenance limits.
