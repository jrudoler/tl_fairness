# Manuscript analyses and reproduction

Use semantic names; LaTeX supplies figure and table numbers. The publication
workflow renders frozen, checksummed results. New computation is a separate,
explicit workflow with a fresh run ID.

## Current manuscript artifacts

| Analysis ID | Rendering target | Manuscript artifact |
|---|---|---|
| `parity_retraining` | `fig_parity_retraining` | `figs/parity_retraining.pdf` |
| `nuisance_early_stopping` | `fig_parity_nuisance_training` | `figs/parity_nuisance_training.pdf` |
| `cmi_conditioning_set` | `fig_cmi_conditioning_set` | `figs/cmi_conditioning_set.pdf` |
| `cmi_estimation` | `fig_cmi_estimation` | `figs/cmi_estimation_{error,coverage}.pdf` |
| `real_data_adult`, `real_data_law` | `table_real_data_inference` | `tables/real_data_inference.tex` |
| `cmi_estimation` | `table_cmi_population_truth` | `tables/cmi_population_truth.tex` |
| `nuisance_early_stopping` | `nuisance_tables` | `tables/nuisance_{population,fitted,controls,sensitivity}.tex` |

Paths in the last column are relative to `paper/`. Results are rendered under
`results/figures/` and `results/data/`, then copied with the same basename.
The real-data and CMI-truth tables are generated inputs; captions stay in the
manuscript. `nuisance_findings.tex` and `nuisance_tables.csv` are reporting aids,
not additional manuscript tables. `stopping_summary.csv` supports numbers in
appendix prose; those prose claims remain editorial text rather than generated
LaTeX. The `nuisance_reference` analysis supplies the fixed-tree control audit
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

The current publication is `manuscript-2026-10-13`, selected from all seven
completed analyses in `data/runs/reproduction-20261012/`. Each ran its saved
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
allow both sets of intervals to be reconstructed without refitting. CMI keeps
the previously verified data-fairness result and has no model-fairness column entry.

The October 14 comparison fits are queued as Slurm jobs 94570 (Adult) and 94571
(Law), under `data/runs/matched-real-data-20261014/`. Dependent job 94588 runs
`analysis/tools/complete_matched_real_data.py` after both succeed. It checks the
approved manuscript draft has not changed, selects only these additional runs,
validates the prediction caches, adds the table dependencies, and rebuilds the
paper. Until selection succeeds, the previous publication remains selected and
buildable. Consult `config/publication.json` for the actual current selection;
the planned publication ID is `manuscript-2026-10-14`. Completion is recorded in
`results/provenance/selection-manuscript-2026-10-14.json` and
`logs/slurm/paper-matched-fairness-94588.log`. Failed prerequisites prevent the
dependent job from running. Manuscript edits made while jobs queue cause a
guarded failure requiring reconciliation, rather than an overwrite.

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
