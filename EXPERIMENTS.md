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
  input by path, SHA-256, and recovery/source record. These inputs live under
  `data/generated/` and `data/raw/`; they are not overwritten by new runs.
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

Recovered retraining replicates reproduce all 405 summary rows. The nuisance
validator checks 118,000 paired-method rows and their manifests/control audits.
The stale local Law cache was deleted on 2026-10-09 after verifying its
corrected replacement against the manuscript table. Its old and replacement
checksums are retained in the [cleanup record](archive/workflow_2026-10-09/stale-law-cache.json). The generated real-data and CMI-truth
tables retain the published values.

Historical CMI estimation and conditioning-set runs saved aggregates without
complete per-replicate records or run manifests. Recovery preserves that
limitation; current source defaults are not evidence of every historical setting.
Some recovered CMI real-data results retain only rounded precision. No new
simulation was substituted for these publication results. New CMI runs save
replicate records and use the explicitly versioned `task-streams-v2` RNG scheme.

To restore an already downloaded recovery bundle (existing conflicting files
cause an error):

```sh
.venv/bin/python analysis/tools/restore_publication.py \
  --bundle data/imported/locust-2026-10-09
```

The separately delivered nuisance bundles must also be present, with the hashes
in `config/publication.json`. A fresh Git clone alone does not contain the data.

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

## Locust

The synchronized Locust checkout is `/home/jrudoler/tl_fairness`. Its publication
inputs, Python environment, figure rendering, and both workflow dry runs have
been checked. A fresh worktree needs its own environment and access to the
untracked data. Manuscript compilation on Locust currently fails because its
TeX installation lacks `LibertinusMath-Regular`; local XeLaTeX compilation passes.

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
workflow. The naive real-data comparison is exploratory and not included in the
manuscript. Historical code and reports remain available without becoming build
dependencies.

[The naming archive](archive/naming_2026-10-09/README.md) preserves changed source
snapshots, old-to-new names, and figure rename checksums. In particular,
`fig9_retraining`, `fig7_trainsize`, `fig8_condset`, and `fig4_cmi` were replaced by
the semantic rendering targets above. Historical reports retain their original
names. The original audit is in [docs/workflow-audit.md](docs/workflow-audit.md);
its follow-up distinguishes resolved issues from historical provenance limits.
