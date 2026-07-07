"""Experiment 5: does "enough training data" rescue the naive baseline? (Setting 3)

A natural objection to the headline result (Exp 2) is:

    "Any sufficiently flexible model, given enough training data, will recover the
     true P(Y|X); then the naive fixed-model + CLT interval is fine -- targeted
     learning buys nothing."

This experiment engages that objection head-on by *decoupling* the ML model's
**training** size from the **evaluation** (test) size. We hold the evaluation set
fixed at a realistic ``--test-size`` and sweep the nuisance-model training size
``--train-sizes``. All three methods see the SAME train split and the SAME
evaluation split each replicate; only the inference differs.

The naive parity CLT (``naive_fixed_model_parity``) estimates
``mean_{g=1} D_hat(X) - mean_{g=0} D_hat(X)`` on the test set and forms a
two-sample-mean Wald interval, treating the fitted ``D_hat`` as if it were the
truth. Its error relative to the data estimand Ψ = E[D(X)|G=1] - E[D(X)|G=0] has
two parts: (a) finite test-set sampling -- which the CLT variance *does* capture,
and (b) model error ``D_hat != D`` -- which it does *not*. The four curves map how
(b) behaves as training data grows:

  * Naive CLT, flexible + DEFAULT gradient booster: coverage *improves* with
    ``n_train`` (0.24 -> ~0.5 here) then plateaus well below nominal. This plateau
    is NOT an irreducible approximation floor -- it is the default learner being
    *underfit* (lr=0.1 x only 100 shrunk rounds have not converged). At a fixed
    ``n_train`` it is curable by more *rounds/capacity*, not more *data*: raising
    n_estimators to 1000 at n_train=5000 drops the bias from ~-0.030 to ~-0.003.
    So "more data" does not rescue a fixed under-capacity learner -- but this is a
    statement about that learner, not about the naive CI in general.
  * Naive CLT, flexible + TUNED gradient booster (``TUNED_GB``): with a
    higher-capacity, better-converged learner the bias (b) shrinks with data and
    coverage *climbs toward nominal* at large ``n_train``. This is the objection's
    real grain of truth: given sample-splitting AND a consistent, well-tuned
    nuisance, the two-sample CLT is asymptotically valid for the parity point
    (the nuisance-estimation variance is O(1/n_train) -> 0). The catch is that you
    cannot tell FROM THE INTERVAL whether you have reached this regime -- the
    default-GB CI above looks just as confident at 0.5 coverage as this one does
    at 0.95.
  * Naive CLT, linear (misspecified): the one genuinely structural failure.
    ``D_hat`` converges to the WRONG limit, so bias (b) is frozen at ~the full
    parity gap; neither more data NOR more capacity moves it -- coverage stays 0.
  * TL one-step, DEFAULT gradient booster (same underfit nuisance as curve 1): the
    EIF correction debiases that finite-sample fit automatically, so coverage
    reaches nominal by ``n_train``~250 and stays calibrated/conservative -- TL
    turns the same nuisance that leaves the naive CLT at ~0.5 into a valid
    interval, with no tuning and nothing to verify. (At a tiny train size the GB
    nuisances are too noisy for the first-order correction to bite, so TL too
    needs *some* data -- just much less, and it never collapses like the linear
    naive.)

Takeaway: the naive CI *can* be made valid (flexible + tuned + enough data +
sample splitting), but its validity is contingent on nuisance quality you cannot
verify from the interval, whereas TL delivers validity from the same imperfect
nuisance for free -- and misspecification (the linear curve) is the one bias that
data and capacity cannot touch.

Usage (smoke):
  PYTHONPATH=. .venv/bin/python experiments/exp5_trainsize_coverage.py \
      --train-sizes 250 1000 --test-size 1000 --reps 20 --n-jobs 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from tlfair.metrics import prob_parity
from tlfair.plotting import configure_matplotlib, FULL_WIDTH
from tlfair.simulations import setting3_draw, setting3_truth
from experiments.baselines import naive_fixed_model_parity

# Display order. All four target the same data estimand (the parity gap); they
# differ only in (i) the fitted model and (ii) how uncertainty is quantified.
METHODS = ["TL one-step (flexible)", "Naive CLT (flexible, default)",
           "Naive CLT (flexible, tuned)", "Naive CLT (linear)"]

# Higher-capacity, better-converged gradient booster: enough rounds (with a
# smaller learning rate) that its fit is no longer dominated by underfitting, so
# its bias shrinks with data rather than stalling at the default learner's floor.
# (Empirically the extra rounds, not the extra depth, do most of the work here --
# see the module docstring.)
TUNED_GB = dict(n_estimators=1000, max_depth=5, learning_rate=0.05)


def _one_rep(n_train, n_test, rng, bernoulli=False):
    """One Setting-3 replicate at a given (n_train, n_test); {method: (est, lo, hi)}."""
    n = n_train + n_test
    # Match the manuscript: both Y and G are Bernoulli (the latter also restores
    # positivity). Tie the group draw to the same flag as the outcome.
    x, g, y, _ = setting3_draw(n, rng, bernoulli=bernoulli, bernoulli_group=bernoulli)
    Xtr, Xte = x[:n_train], x[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    gtr, gte = g[:n_train], g[n_train:]
    seed = int(rng.integers(0, 2**31 - 1))  # reproducible GB fits

    out = {}
    # --- TL one-step, DEFAULT gradient booster: calibrated at any training size.
    # Deliberately the SAME default (underfit) learner as the naive-default curve
    # below, so the figure shows TL turning that nuisance into a valid interval.
    est, (lo, hi) = prob_parity(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        group_train=gtr, group_test=gte,
        outcome=GradientBoostingClassifier(random_state=seed),
        propensity=GradientBoostingClassifier(random_state=seed + 1),
    )
    out["TL one-step (flexible)"] = (est, lo, hi)

    # --- Naive fixed-model + two-sample CLT, same train/eval splits ---
    # (these helpers return (est, (lo, hi)); flatten to (est, lo, hi))
    # Default GB: underfit at these hyperparameters, so its bias plateaus with n.
    est, (lo, hi) = naive_fixed_model_parity(
        GradientBoostingClassifier(random_state=seed), Xtr, ytr, Xte, gte)
    out["Naive CLT (flexible, default)"] = (est, lo, hi)
    # Tuned GB: higher capacity / better converged, so its bias shrinks with data
    # and coverage climbs toward nominal -- the objection's real grain of truth.
    est, (lo, hi) = naive_fixed_model_parity(
        GradientBoostingClassifier(random_state=seed, **TUNED_GB), Xtr, ytr, Xte, gte)
    out["Naive CLT (flexible, tuned)"] = (est, lo, hi)
    # Linear logistic on raw x: misspecified (the truth is logistic in x**2), so
    # D_hat converges to the wrong limit -- bias persists at every train size.
    est, (lo, hi) = naive_fixed_model_parity(
        LogisticRegression(max_iter=1000), Xtr, ytr, Xte, gte)
    out["Naive CLT (linear)"] = (est, lo, hi)
    return out


def _aggregate(results, method, truth, n_train, n_test):
    ests = np.array([r[method][0] for r in results])
    los = np.array([r[method][1] for r in results])
    his = np.array([r[method][2] for r in results])
    covered = (los <= truth) & (truth <= his)
    return {
        "method": method,
        "train_size": n_train,
        "test_size": n_test,
        "coverage": float(np.mean(covered)),
        "bias": float(np.mean(ests) - truth),
        "mean_ci_width": float(np.mean(his - los)),
        "mean_estimate": float(np.mean(ests)),
        "truth": truth,
    }


def run(train_sizes, test_size, reps, seed, n_jobs, truth_n, bernoulli=False):
    rng = np.random.default_rng(seed)
    parity_truth = setting3_truth(truth_n, rng, bernoulli_group=bernoulli)  # group law must match data
    print(f"Setting-3 probabilistic-parity truth = {parity_truth:.5f}"
          f"  [outcome={'Bernoulli' if bernoulli else 'deterministic'}, "
          f"test_size={test_size}]", flush=True)

    rows = []
    for nt in train_sizes:
        children = rng.spawn(reps)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_rep)(nt, test_size, c, bernoulli) for c in children)
        for m in METHODS:
            rows.append(_aggregate(results, m, parity_truth, nt, test_size))
        cov = {r["method"]: r["coverage"] for r in rows[-len(METHODS):]}
        print(f"  n_train={nt}: " + ", ".join(f"{m}={cov[m]:.3f}" for m in METHODS),
              flush=True)
    return pd.DataFrame(rows)


def plot(df, output):
    import matplotlib.pyplot as plt
    import seaborn as sns
    configure_matplotlib()
    test_size = int(df["test_size"].iloc[0])
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.55, FULL_WIDTH * 0.42))
    sns.lineplot(data=df, x="train_size", y="coverage", hue="method",
                 marker="o", ax=ax)
    ax.axhline(0.95, ls="--", color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"Nuisance-model training size (eval set fixed at n={test_size})")
    ax.set_ylabel("95% CI coverage")
    ax.set_ylim(0, 1.02)
    ax.legend(title=None, fontsize=7)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sizes", nargs="+", type=int,
                        default=[100, 250, 500, 1000, 2500, 5000, 10000])
    parser.add_argument("--test-size", type=int, default=2000,
                        help="evaluation-set size, held FIXED while train size sweeps")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--truth-n", type=int, default=10_000_000)
    parser.add_argument("--deterministic", action="store_true",
                        help="use the deterministic Bayes-decision outcome instead of "
                             "the default Bernoulli draw (Bernoulli is canonical; see "
                             "experiments/exp2_misspec_coverage.py / audit_glm.py).")
    parser.add_argument("--output", default="experiments/out/exp5_trainsize_coverage.csv")
    parser.add_argument("--figure", default="experiments/out/exp5_trainsize_coverage.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = run(args.train_sizes, args.test_size, args.reps, args.seed, args.n_jobs,
             args.truth_n, bernoulli=not args.deterministic)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)
    print(df.to_string(index=False), flush=True)
    plot(df, args.figure)


if __name__ == "__main__":
    main()
