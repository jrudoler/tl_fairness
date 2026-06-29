# Baseline-coverage experiments (exploratory)

Head-to-head **validity** comparisons of the targeted-learning (TL) data-fairness
estimators against the three naive alternatives ("straw men"). These are
standalone scripts, **not** wired into the Snakemake pipeline; outputs land in
`experiments/out/` (gitignored). Promote to the pipeline once the story holds
(see the plan / `Promotion path`).

The repo already shows TL attains nominal coverage (Figs 2/4/6) and that the
naive *variance* is too small (Fig 3). What was missing — and what these add — is
a head-to-head **coverage / calibration** comparison against the named baselines
on known-truth DGPs.

## Run

```bash
# smoke (seconds)
PYTHONPATH=. .venv/bin/python experiments/exp1_parity_coverage.py --sizes 250 500 --reps 20 --n-jobs 4
PYTHONPATH=. .venv/bin/python experiments/exp2_misspec_coverage.py --sizes 250 500 --reps 20 --n-jobs 4
PYTHONPATH=. .venv/bin/python experiments/exp3_cmi_permutation.py --weights 0 1 --sizes 1000 --reps 20 --n-perm 100 --n-jobs 4

# paper-scale (minutes on a multi-core node)
PYTHONPATH=. .venv/bin/python experiments/exp1_parity_coverage.py --reps 500 --n-jobs 24
PYTHONPATH=. .venv/bin/python experiments/exp2_misspec_coverage.py --reps 300 --n-jobs 24
PYTHONPATH=. .venv/bin/python experiments/exp3_cmi_permutation.py --reps 200 --n-perm 200 --n-jobs 24
```

Each writes a tidy CSV + a PNG to `experiments/out/` and prints a summary table.
Seeding uses `np.random.default_rng(seed)` + `rng.spawn(reps)`, so results are
**invariant to `--n-jobs`**.

### Bernoulli outcomes are canonical; `--deterministic` opts out (Exp 1 & 2)

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
`False`, so any caller that does not opt in — `sim_tmle`/Figure 6 and the unit
tests — is bit-identical to before.

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
  so the MLE diverges (a degenerate fit). Requires `statsmodels`
  (`uv pip install statsmodels`).

- **Exp 1 — Setting 1, probabilistic parity (well-specified, parametric).** The
  regime most favourable to the straw man. The naive CLT is ~calibrated here and
  TL is valid but conservative; the naive interval starts slipping under mild
  ("linear") misspecification. Point: the naive "model fairness" CI and the TL
  "data fairness" CI nearly coincide *only* when the model is easy to estimate.

- **Exp 2 — Setting 3, misspecification (the headline).** With flexible/
  misspecified models the naive CLT under-covers parity (and *worsens* with n:
  ~0.77 → ~0.57), a linear model collapses to 0 coverage, and the GLM "adjusted
  effect" — which targets a *different* estimand (structurally 0 here, since the
  group affects the outcome only through X) — looks fine when well specified but
  collapses under misspecification (spurious effect, tiny CI — and the sandwich
  SE confirms this is bias, not an optimistic variance). The well-specified GLM
  is itself degenerate here because the noiseless DGP is separable. TL stays
  calibrated throughout.

- **Exp 3 — CMI conditional independence (permutation answers the wrong
  question).** At c=0 (conditional independence, but marginal dependence through
  the shared confounder Z) the global permutation test has ~100% Type-I error;
  TL conditions on Z fully and is calibrated/conservative; the stratified
  permutation only conditions approximately (coarse bins) and partly recovers
  calibration. Power rises with the dependence strength c.

## Reused from `tlfair/`

Setting-1 DGP/truth (`_setting1_draw`, `parity_ground_truth`), Setting-3 DGP/truth
(`setting3_draw`, `setting3_truth` — lifted here from `analysis/sim_robust/run.py`),
the TL estimators (`metrics.prob_parity`, `metrics.cmi`), the CMI DGP, and the
plotting defaults (`plotting.configure_matplotlib`).
