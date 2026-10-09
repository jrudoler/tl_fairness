# Retired workflow targets — 2026-10-09

`Snakefile` is an unchanged snapshot of the workflow before the manuscript
inventory audit. It is **not included by the active workflow**. It preserves
the old rule definitions and output names, not a self-contained frozen software
environment. Consult Git history and the original run manifests for exact code
and runtime provenance. Do not use its default target to reproduce the current
manuscript.

The following rules were removed from the active `Snakefile`:

| Rules | Status / reason |
|---|---|
| `sim_parity`, `fig1_parity`, `fig3_variance` | Historical thresholded-parity sweep and probabilistic-parity variance comparison; neither figure is included. |
| `sim_robust`, `fig2_robust` | Historical robustness heatmap; replaced in the manuscript by the matched nuisance simulations and their controls. |
| `sim_tmle`, `fig6_tmle` | TMLE comparison; no current figure or table uses it. Previously still in `all` and the paper copy map. |
| `sim_nuisance_tuning` | Deferred cross-validation extension. |
| `sim_nuisance_matched_rate` | Intermediate fixed-tree learning-rate revision, superseded by early stopping. |

The manuscript workflow now requests only the five reported real-data metrics,
and the generated manuscript table omits the three TMLE comparison rows.
The standalone scripts retain their optional TMLE functionality for historical
work. The manuscript validator excludes the retired robustness/TMLE outputs
and additionally runs the existing early-stopping validation.

Source scripts, saved results, and historical manuscript assets were left in
place. Moving those scripts would invalidate recorded paths/source hashes or
break imports in retained audits; deleting results would destroy provenance.
This is an archive of workflow participation, not a purge of scientific records.

## Other analyses outside the manuscript workflow

- `experiments/exp1_parity_coverage.py`, `exp2_misspec_coverage.py`,
  `exp3_cmi_permutation.py`, and `exp5_trainsize_coverage.py`: historical standalone
  experiments; no current manuscript figure is generated from them.
- `analysis/fig5_importance/`, `tlfair.metrics.perm_importance`: retired feature
  importance. Already disconnected/commented out before this audit.
- `slurm/fill_tmle.{py,sbatch}`: historical utilities for the retired TMLE rows;
  not part of the active DAG. Do not use on current manuscript inputs.
- `analysis/eif_regeneration/`, `slurm/eif_regeneration.sbatch`: correction and
  provenance audits. Retain as records of how historical results were repaired.
- `experiments/audit_glm.py`, `analysis/nuisance_tables/diagnose_bias.py`: diagnostic
  checks, not current manuscript artifact producers.
- `notebooks/`: historical exploratory work, already documented as such.
- Old `paper/figs/{asymptotic,naive_var,robust_coverage,tmle_coverage,cmi_tmle_*}`
  PDFs/PNGs and PNG duplicates of active PDF figures are unused by current TeX.
  Their physical relocation is deferred to the proposed manuscript asset cleanup.

## Dependencies that must remain available

- `sim_trainsize` / `experiments/nuisance_training.py` generates the original
  fixed-tree reference. Early-stopping collection reads its population values
  and replicates; validation compares unchanged controls and runtime/source
  metadata. Its results are **supporting provenance**, not the current Figure 2.
- `experiments/exp4_feature_selection.py` is active through `sim_condset` and
  `fig_condset`, despite its exploratory-looking name.
- `experiments/exp6_retraining.py` supplies Figure 1 and a shared interval routine.
- `experiments/baselines.py` is imported by the active conditioning-set analysis.
- `analysis/naive_real_data/` is a pending manuscript comparison, with cached
  predictions and provenance. It is not currently included, but should not be
  called superseded: the request to compare naive intervals remains relevant.
- `nuisance_findings.tex` is a generated numerical summary, not an included TeX
  section. Retained as a reporting aid pending a separate numerical-claims design.

See [the full audit](../../docs/workflow-audit.md) for active artifact lineage,
proposed semantic names, and remaining reproducibility work.
