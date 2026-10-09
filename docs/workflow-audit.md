# Manuscript workflow and provenance audit

## Follow-up completed 2026-10-09

The inventory and findings below are the original audit, retained as history.
Current commands and authoritative paths are in [EXPERIMENTS.md](../EXPERIMENTS.md).

- Recovered and checksum-verified 96 Locust files (1.36 GB), including corrected
  Adult/Law caches, CMI summaries/truth, conditioning-set summaries, and the full
  retraining run. Quarantined the stale Law cache. Remote shared-data source/shard
  snapshots remain intact.
- Applied semantic figure/analysis names and removed inactive rules from the
  publication DAG; original sources/workflow are archived.
- Added publication input hashes, artifact lineage, explicit new-run configurations,
  immutable dataset URLs, and source/config/runtime/output manifests.
- Separated new computation from validation/rendering/delivery/compilation. Missing
  publication data fail without launching simulation. Rendering uses tracked style;
  compilation uses XeLaTeX. Rendering no longer appends to the original run log.
- Connected generated real-data and CMI-truth table bodies to the manuscript.
  Their formatted values match the recovered publication tables.
- Unified serial/parallel CMI RNG streams and separated truth/coverage/comparison
  task streams for new runs; preserved historical publication results. New CMI
  runs save per-replicate records. Worker invariance is tested.
- Reconstructed all 405 retraining summary rows from saved replicates and passed
  the nuisance validation of 118,000 paired-method rows. Rendering and table
  delivery use the verified publication registry.

Remaining historical limitations: legacy CMI runs lack full replicate/manifests;
recovery cannot manufacture that provenance. Some historical real-data CMI values
have only rounded precision. Numbers in appendix prose remain manually written,
although their source diagnostics are registered. The real-data implementation
uses a histogram gradient booster for CMI outcomes and SuperLearners for the other
reported outcomes; the manuscript's broader learner wording merits a separate
editorial discussion. None of these limitations is silently resolved by rerunning.

Reviewed 2026-10-09 against the active `paper/main.tex` inputs, included figures
and tables, and the current auxiliary-file numbering. Figure/table numbers here
are descriptive; they must not become permanent output identifiers.

This is an audit of the local checkout and source, not a claim that missing
artifacts are also absent from the cluster. No simulation was rerun, no published
PDF was overwritten, and no manuscript prose was edited. The limited implemented
changes remove retired workflow participation; the broader migration below is
proposed for discussion.

## Original artifact inventory

| Manuscript artifact | Current producer and inputs | Current result / manuscript copy | Proposed stable identifier |
|---|---|---|---|
| Figure 1: independent retraining | `sim_retraining` → `experiments/exp6_retraining.py`; `retraining_dense/summary.csv` → `fig9_retraining` | `fig9_retraining.pdf` → `paper/figs/retraining.pdf` | `parity_retraining` |
| Figure 2: nuisance learning | `sim_trainsize` supplies reference data; `sim_nuisance_early_stopping` → `experiments/nuisance_early_stopping.py`; `nuisance_early_stopping/summary.csv` → `analysis/fig_trainsize/run.py` | `fig7_trainsize_coverage.pdf` → `paper/figs/trainsize_coverage.pdf` | `parity_nuisance_training` |
| Figure 3: conditioning set | `sim_condset` → `experiments/exp4_feature_selection.py`; `condset.csv` → `analysis/fig_condset/run.py` | `fig8_conditioning_set.pdf` → `paper/figs/conditioning_set.pdf` | `cmi_conditioning_set` |
| Figure 4: CMI comparison/coverage, appendix | `sim_cmi` → `tlfair/cmi_sim.py`; `cmi_compare.csv`, `cmi_coverage.csv`, `truth_dict.pkl` → `analysis/fig4_cmi/run.py` | `fig4_cmi_{error,coverage}.pdf` → `paper/figs/cmi_{error,coverage}.pdf` | `cmi_estimation_{error,coverage}` |
| Table 1: Adult/Law inference | downloads → `analysis/analyze_{adult,law}/run.py` → result pickles → `table1_inference` | `table1_inference.{csv,tex}`; manuscript values are manually embedded in `sections/data_analysis.tex` | `real_data_inference` |
| Tables 2–5: population, fitted nuisances, controls, sensitivity | early-stopping `population.csv`, `summary.csv`, `paired_comparisons.csv` → `analysis/nuisance_tables/run.py` | `results/data/nuisance_*.tex` → `paper/tables/nuisance_*.tex`, included by appendix | `nuisance_{population,fitted,controls,sensitivity}` |
| Table 6: CMI truth, appendix | `sim_cmi` → `truth_dict.pkl` → `table2_cmi_truth` | `table2_cmi_truth.{csv,tex}`; manuscript values manually embedded in `sections/appendix_cmi.tex` | `cmi_population_truth` |
| Early-stopping numerical diagnostics in appendix prose | saved replicate tree counts/probability errors → `nuisance_stopping_diagnostics` | `nuisance_early_stopping/stopping_summary.csv`; values copied into prose | `nuisance_stopping_diagnostics` |

Intermediate paths in the table are under `data/generated/`; result PDF
basenames are under `results/figures/` unless otherwise noted. Each manuscript
figure should ultimately use the same semantic basename as its source figure.
`trainsize_coverage` is incomplete as a name because the figure also reports bias.

## Configuration map

These are the current source defaults/overrides, not reconstructed manifests
for missing historical runs.

- **Parity retraining:** four-cell saturated probability estimates with half-count
  smoothing; outcome effects `[0, .2, .6]`, group association `.6`; training sizes
  `[100,150,225,350,500,750,1000,1500,2000]`; evaluation sizes
  `[250,500,1000,2000,4000]`; 1000 training replicates, 50 evaluations per fit;
  seed `20260909`. Two-sided 95% intervals; Monte Carlo errors cluster by fit.
  The script writes raw replicates, summaries, `config.json`, log, and PDF;
  Snakemake declares only raw replicates and summary for the simulation step.
- **Nuisance training:** five normal predictors; primary nonlinearity `.5`,
  sensitivity `0` and `.75`, separate squared-feature stress population;
  training sizes `[100,250,500,1000,2500,5000,10000]`, controls/sensitivity at
  `[500,2500,10000]`; evaluation size 2000; 1000 replicates; seed `20261005`.
  Boosters: depth/tree caps `(3,100)` and `(5,1000)`, learning rate `.1`, holdout
  `.2`, patience 20, tolerance `1e-4`. Truth uses replicated Sobol integration.
  Source, runtime, seed, learner, and replicate hashes are recorded. The original
  fixed-tree run remains a reference dependency for control audits.
- **CMI conditioning set:** pipeline uses 30 features, 5 confounders, 8 outcome
  predictors, signal `1.5`, sizes `[2500,5000]`, 200 replicates, coefficient seed
  `0`, data seed `123`. The standalone experiment defaults differ: signal `1.0`
  and sizes `[1000,2500]`. The displayed lower bound and Wald rejection both use
  `1.645 SE`; the diagnostic rule is not calibrated at conditional independence.
  The CSV is aggregated; the pipeline does not save per-replicate results or a
  run manifest.
- **CMI estimation:** coupling weights `0:.5:4`; sizes
  `[500,750,1000,1750,2500,3750,5000,7500,10000]`; 100 coverage repetitions,
  10 TL/KNN comparison repetitions; one million truth draws; seed `123`.
  Conditional CMI truth is now the default. Coverage uses two-sided intervals.
  Outputs are aggregates, truth pickle, and timings, with no full run manifest.
- **Real data:** Adult reconstruction from folktables and cleaned Law data from
  fairness_dataset; seed `123`, 60/40 split. Non-CMI outcomes and group models
  use SuperLearner; CMI uses a histogram gradient booster for the outcome model.
  Thus the manuscript's blanket statement about SuperLearners is not a complete
  description of the implementation. The workflow now requests the five reported
  metrics explicitly. The ordinary analysis pickles lack a run manifest or cached
  prediction/split records; the separate naive-interval script has stronger
  provenance and cached outcome predictions.

## Findings requiring follow-up

### 1. Recover and identify the exact inputs behind current manuscript artifacts

Only the nuisance-training figure has its generating summaries and result PDF
available locally. The local checkout lacks `retraining_dense/`, `condset.csv`,
`cmi_coverage.csv`, `cmi_compare.csv`, and `truth_dict.pkl`, while the corresponding
manuscript PDFs exist. Recover the exact saved run bundles from the compute host
before renaming or rebuilding these outputs. A fresh simulation would be a new
run, not proof of the old PDF's lineage.

The conditioning-set PDF was previously adjusted directly to match the one-sided
display while its CSV was unavailable. Its source plotting code now uses that
display, but reproducing the delivered PDF still requires the original data and
a recorded rendering step. Record this intervention explicitly in its manifest.

The local `law_results.pkl` contains obsolete intervals: thresholded parity is
`0.22624 (0.17447, 0.27801)`, unlike the manuscript's `0.23 (0.20, 0.25)`.
Probabilistic parity is `0.19168 (0.14276, 0.24059)`, unlike `0.19 (0.17, 0.21)`.
Do not promote this cache into the current paper. Recover the corrected analysis
bundle documented in `docs/eif-regeneration.md`, and quarantine stale caches
with their original metadata before any rerun. The local Adult results pickle
is absent.

### 2. Complete the manuscript dependency graph

`snakemake paper` does not depend on either generated real-data or CMI-truth
table, because both tables are typed directly into section files. Changes to
those analyses therefore do not propagate to the manuscript. Proposed change:
generate table-body inputs and keep editorial captions in TeX; separately record
or generate numerical claims in prose. This requires a discussed manuscript edit.

The nuisance table chain is substantially better, but `nuisance_fitted.tex` in
`results/data/` differs from the manuscript copy in caption wording only. The
renderer already contains the revised wording; its source is not a declared
input, so Snakemake can miss the need to rerender. `nuisance_findings.tex` is
generated/copied but not included; classify it as a reporting aid rather than a
manuscript table. `stopping_summary.csv` supports numerical prose, not the table
renderer despite appearing in that rule's inputs.

### 3. Make configuration and source dependencies explicit

Most rules declare data dependencies only; editing their Python entrypoint,
shared estimator, or plotting style will not reliably invalidate outputs.
`fig9_retraining` declares its script and `tlfair/plotting.py`, but even it omits
the style file. Add each relevant entrypoint/module/configuration as an input
and use named data inputs so shell commands do not accidentally receive source
paths as data arguments.

Use versioned configurations per analysis, rather than defaults spread across
the Snakefile, argument parsers, experiment modules, and Slurm scripts. Keep
scientific settings separate from resources. Archive run-specific resolved
configuration alongside results, including the explicit conditional-CMI target,
interval/test convention, train/evaluation sizes, learner settings, and seeds.

### 4. CMI currently depends on execution resources

In `tlfair/cmi_sim.py`, `n_jobs=1` draws sequentially from the parent RNG, whereas
parallel coverage and comparison spawn child RNGs. Thus switching between serial
and parallel changes the simulated draws. `sim_cmi` reuses the parent RNG across
truth and coverage calculations, so this can also affect subsequent truth draws.
The comparison memory cap can change whether a configuration takes the serial
or parallel path. Fix by assigning deterministic streams to scientific task IDs
and using the same tasks in both execution modes; treat this as a versioned
simulation change and preserve the existing publication run. Do not silently
regenerate and overwrite its results.

### 5. Standardize provenance and validation

Use the early-stopping run as the starting pattern: a run manifest should record
analysis/run ID, parent run IDs, resolved settings, seeds, source hashes and Git
revision/dirty state, raw/input checksums, runtime/lockfile versions, commands,
and output checksums. Keep per-replicate results sufficient to reconstruct every
summary and interval claim. Record simulation and collection/rendering runtimes
separately; they already differ in the nuisance run.

Pin external dataset URLs to immutable revisions and record retrieval URLs,
checksums, preprocessing, split indices, and fitted-model settings. The current
downloaders fetch moving `main` branches and reserialize CSVs without a manifest.

The existing generic validator checks schemas and finite values, not whether
results came from the current estimator. It would accept a structurally valid
stale Law cache. Retraining and conditioning-set outputs also need dedicated
validation and manuscript-copy checks. `validate` now includes the existing
stronger nuisance validator and excludes retired experiments, but is not yet a
complete provenance gate. Ideally validation reads existing bundles and fails
clearly when absent, rather than implicitly launching expensive simulations.

### 6. Separate recomputation, rendering, delivery, and compilation

Currently `snakemake paper` can launch missing simulations before compiling.
Its compile command specifies `latexmk -pdf`, while the recent successful build
uses XeLaTeX. Confirm the project engine and specify it consistently. The rule
also watches excluded appendix files via a broad glob and omits some TeX build
inputs. Prefer separate explicit simulation, collection, validation, rendering,
delivery, and compile targets.

The plotting helper prefers `~/clean-figs.mplstyle` over the repository copy,
so appearance can change by machine. Use the tracked style by default and record
intentional overrides. Neither code changes nor style changes should require
refitting nuisance models merely to redraw a figure.

### 7. Replace ordinal names and outdated documentation together

Use `analysis/<semantic_id>/`, `data/generated/<semantic_id>/<run_id>/`, and
`results/figures/<semantic_id>.pdf`. Include a run/config ID when variants coexist;
reserve figure numbers for LaTeX. Do not mechanically rename frozen source files:
first preserve their snapshots/hashes and update imports, tests, Slurm entrypoints,
default paths, rule names, and the artifact registry together.

`EXPERIMENTS.md` still describes retired figures, permutation-importance compute,
and the old unconditional truth option. `experiments/README.md` and
`docs/nuisance-training-revision.md` mix current early-stopping instructions with
superseded fixed-tree/CV instructions. Separate a concise current reproduction
guide from dated historical records. The archive index documents which analyses
are retired and which apparently historical modules are still dependencies.

## Checks performed and implemented scope

- Archived the pre-audit workflow and removed nine inactive rule definitions.
  TMLE is no longer a default figure, manuscript-copy dependency, or requested
  real-data metric; manuscript table rendering can omit TMLE rows.
- Kept the original nuisance reference dependency and all historical data/source
  files. No manuscript assets were moved or edited.
- Verified the early-stopping delivery hashes for all eight recorded intermediate
  outputs and the Figure 2 PDF; the local and manuscript Figure 2 PDFs match.
- Ran the nuisance validator successfully: 118,000 paired-method rows, 118
  summaries, and 16 configurations.
- Clean-source, no-artifact dry runs of `all`, `paper`, and `validate` succeed.
  Retired targets are absent from the active rule list and planned jobs.
- Before edits, the local `all` dry run stopped at a protected, stale Law cache.
  The local Figure 2 dry run considered results current but reported missing
  Snakemake metadata; external delivery manifests supply information the DAG
  itself does not have. No permission bits were changed and no expensive jobs ran.

Recommended order: recover/quarantine artifact bundles; agree on semantic names
and the registry; introduce explicit configurations/manifests and source inputs;
connect the remaining manuscript tables; then make resource-independent RNG and
rendering/build changes under separately identified runs.
