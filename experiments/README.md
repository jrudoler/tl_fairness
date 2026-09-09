# Baseline-coverage experiments (exploratory)

## September 2026 audit: independent retraining

**Start with [the manuscript/experiment review](../docs/fairness-retraining-review.md)
and `exp6_retraining.py`.** Experiments 1, 2, and 5 already redraw training data,
but had no paired population-null/alternative rejection analysis. Experiment 6
adds exact null and alternative truths, 1000 independent training samples per
configuration, 50 fresh evaluations per fitted model across five evaluation sizes, own-model vs data-target
coverage, Type-I error/power, paired disagreement rates, and a nested variance
decomposition. Monte Carlo errors cluster by training dataset.

The audit corrected group centering in the probabilistic EIF variance (one-step,
NumPy TMLE, and JAX TMLE), plus group sample-size denominators in the Figure 3
baseline. The full regeneration is documented in
[the correction report](../docs/eif-regeneration.md), including the additional
thresholded-metric correction and every change in interpretation. Larger TL
intervals alone do not establish validity or higher power. The corrected
experiments show small-sample undercoverage, limitations under nuisance
misspecification, and no uniform advantage over tuned model-based inference.

```bash
# From the repository root; uv uses the existing project environment.
uv run --no-sync .venv/bin/python experiments/exp6_retraining.py --reps 50 --evaluations 3
uv run --no-sync .venv/bin/python experiments/exp6_retraining.py
# Independence control: a fitted group-blind rule cannot acquire population disparity.
uv run --no-sync .venv/bin/python experiments/exp6_retraining.py --effects 0 --association 0 \
    --output-dir experiments/out/retraining_independent_dense
# Integrated paper-scale run (also builds the figure).
uv run --no-sync snakemake sim_retraining --cores 1
# Redraw the manuscript figures from saved summaries; no simulation rerun.
uv run --no-sync .venv/bin/python experiments/exp6_retraining.py --plot-only \
    --output-dir data/generated/retraining_dense --figure results/figures/fig9_retraining.pdf
```

Output paths are resolved relative to the repository even when the script is
called from another directory. The standalone output directory contains
`replicates.csv.gz`, `summary.csv`, `config.json`, `run.log`, and `retraining.pdf`.
The default grid uses nine training sizes (100, 150, 225, 350, 500, 750, 1000,
1500, 2000) and five evaluation sizes (250, 500, 1000, 2000, 4000).
The pipeline writes data to `data/generated/retraining_dense/` and the figure to
`results/figures/fig9_retraining.pdf`; `snakemake paper` copies it into the paper.
The four-cell learner is saturated and deterministic given training data, so
this experiment isolates training sampling noise without misspecification or
random optimizer effects. Evaluation outcomes are used by TL's correction;
model fairness only needs evaluation predictions and group labels.

## Earlier experiments

Head-to-head **validity** comparisons of the targeted-learning (TL) data-fairness
estimators against the three naive alternatives ("straw men"). These are
standalone scripts, **not** wired into the Snakemake pipeline; outputs land in
`experiments/out/` (gitignored). Promote to the pipeline once the story holds
(see the plan / `Promotion path`).

These experiments compare coverage and calibration against named baselines
on known-truth DGPs. Corrected EIF intervals expose limitations that were
hidden by inflated historical standard errors; see the correction report.

## Run

```bash
# smoke (seconds)
PYTHONPATH=. .venv/bin/python experiments/exp1_parity_coverage.py --sizes 250 500 --reps 20 --n-jobs 4
PYTHONPATH=. .venv/bin/python experiments/exp2_misspec_coverage.py --sizes 250 500 --reps 20 --n-jobs 4
PYTHONPATH=. .venv/bin/python experiments/exp3_cmi_permutation.py --weights 0 1 --sizes 1000 --reps 20 --n-perm 100 --n-jobs 4
PYTHONPATH=. .venv/bin/python experiments/exp4_feature_selection.py --p 20 --n-confounders 3 --n-outcome 5 --sizes 1000 --reps 20 --n-jobs 4
PYTHONPATH=. .venv/bin/python experiments/exp5_trainsize_coverage.py --train-sizes 250 1000 --test-size 1000 --reps 20 --n-jobs 4

# paper-scale (minutes on a multi-core node)
PYTHONPATH=. .venv/bin/python experiments/exp1_parity_coverage.py --reps 500 --n-jobs 24
PYTHONPATH=. .venv/bin/python experiments/exp2_misspec_coverage.py --reps 300 --n-jobs 24
PYTHONPATH=. .venv/bin/python experiments/exp3_cmi_permutation.py --reps 200 --n-perm 200 --n-jobs 24
PYTHONPATH=. .venv/bin/python experiments/exp4_feature_selection.py --reps 200 --signal 1.5 --sizes 2500 5000 --n-jobs 24
PYTHONPATH=. .venv/bin/python experiments/exp5_trainsize_coverage.py --reps 300 --n-jobs 24
```

Each writes a tidy CSV + a PNG to `experiments/out/` and prints a summary table.
Seeding uses `np.random.default_rng(seed)` + `rng.spawn(reps)`, so results are
**invariant to `--n-jobs`**.

### Bernoulli outcomes are canonical; `--deterministic` opts out (Exp 1, 2 & 5)

The original Settings 1/3 realised the outcome as the deterministic Bayes
decision `y = 1{y_probs > 0.5}`, which makes the data noiseless: `Var(Y|X)=0` and
the well-specified logistic GLM is perfectly **separable** (a degenerate MLE; see
`audit_glm.py`). The canonical DGP now draws `y ~ Bernoulli(y_probs)` so
`P(Y=1|X) = y_probs` genuinely holds, the data are non-separable, and the
well-specified GLM becomes a fair, nominal-coverage baseline (Exp 2 panel B:
~0.60 deterministic -> ~0.95 Bernoulli). The estimand *truths* are unchanged
(they are functions of `y_probs`, not the realised `y`), so coverage numbers stay
comparable. Pass `--deterministic` to reproduce the old noiseless behaviour.

This is wired into the pipeline sims too: `analysis/sim_parity/run.py` and
`analysis/sim_robust/run.py` pass `bernoulli=True` (Figures 1–3). The
`bernoulli` keyword on the DGP draws (`tlfair/simulations.py`) defaults to
`False`; the current manuscript simulation entrypoints, including `sim_tmle`,
explicitly opt into Bernoulli outcomes.

## What each shows

- **`baselines.py`** — the three naive estimators/tests: `naive_fixed_model_parity`
  (fixed model + two-sample CLT), `glm_ame_parity` (logistic GLM average marginal
  effect; delta-method Wald CI with a Huber-White **sandwich (HC0)** SE by
  default, or `cov_type="model"` for the inverse-information SE), and
  `permutation_mi_test` / `permutation_cmi_test` (global vs. Z-stratified
  permutation).

- **`audit_glm.py`** — cross-checks the hand-rolled GLM (coefficients, model SE,
  HC0 SE, discrete AME) against `statsmodels` (matches to ~1e-5), and shows that
  the noiseless DGP makes the *well-specified* feature set perfectly separable,
  so the MLE diverges (a degenerate fit). Requires the `experiments` extra
  (`uv sync --extra experiments`).

- **Exp 1 — Setting 1, probabilistic parity.** In the corrected rerun, TL
  coverage is 93.0–95.8% and naive fixed-model coverage is 92.2–95.2% (500
  replicates per configuration). TL intervals are modestly wider, but neither
  uniform conservatism nor uniformly superior coverage is supported.

- **Exp 2 — Setting 3, misspecification.** TL parity coverage is 91.0–95.3%,
  with undercoverage at small sizes. Correctly specified standardization with
  CLT or bootstrap intervals can have comparable coverage. Misspecified linear
  standardization has zero parity coverage; misspecified adjusted-effect GLMs
  continue to reject their true zero effect. The adjusted-effect and parity
  panels concern different estimands, not interchangeable fairness criteria.

- **Exp 3 — CMI conditional independence (permutation answers the wrong
  question).** At c=0 (conditional independence, but marginal dependence through
  the shared confounder Z) the global permutation test has ~100% Type-I error;
  TL conditions on Z fully and is calibrated/conservative; the stratified
  permutation only conditions approximately (coarse bins) and partly recovers
  calibration. Power rises with the dependence strength c.

- **Exp 4 — feature-set selection decides the data-fairness verdict.** A
  high-dimensional, random-coefficient DGP with genuine confounders `Z` (a common
  cause of both `G` and `Y`, so `Y ⊥ G | Z` holds *exactly*) plus outcome-only
  predictors and noise. `G` and `Y` are marginally dependent through the shared
  `Z` (a nonzero marginal `MI(Y;G)` is printed). Sweeping how many confounders are
  in the conditioning set — outcome predictors + noise always included — the TL CMI
  estimate falls monotonically to ~0 and its one-sided Wald test stops flagging:
  with **all** of `Z` the setup **passes** (flag rate ≈ α, calibrated), but drop
  the confounders and it is **flagged** as unfair even though you adjusted for many
  *predictive* features. Point: predictive power is not fairness sufficiency — the
  minimal sufficient adjustment set is the confounders, and the fairness verdict is
  only as good as the conditioning set. Complement to Exp 3 (which fixed the
  feature set and varied the method; Exp 4 fixes the TL CMI method and varies the
  feature set). The confounded DGP is defined inline in the script; promote it to
  `tlfair/simulations.py` if the story graduates into the pipeline. Tune `--signal`
  / `--reps` / `--sizes` to sharpen the with-`Z`/without-`Z` flip.

- **Exp 5 — Setting 3, training size.** The corrected TL interval under-covers
  at small training sizes: coverage is 50.0%, 79.3%, 86.0%, and 91.7% at
  training sizes 100, 250, 500, and 1000. Coverage approaches 95% with thousands
  of training observations. TL improves substantially over default-booster
  fixed-model intervals, but tuned-booster intervals can have comparable
  coverage at large sizes. The linear model remains misspecified with zero
  coverage. The former claim of validity by training size 250 was caused by
  inflated EIF variance and has been removed.

## Reused from `tlfair/`

Setting-1 DGP/truth (`_setting1_draw`, `parity_ground_truth`), Setting-3 DGP/truth
(`setting3_draw`, `setting3_truth` — lifted here from `analysis/sim_robust/run.py`),
the TL estimators (`metrics.prob_parity`, `metrics.cmi`), the CMI DGP, and the
plotting defaults (`plotting.configure_matplotlib`). Exp 4 additionally reuses
`baselines._sigmoid` (loky-safe `expit`) and `baselines._binary_mi`.
