"""Experiment 5: training size and fixed-model inference in Setting 3.

Hold the evaluation sample at 2000 and compare TL one-step with fixed-model
intervals from default boosting, tuned boosting, and linear logistic regression.
All methods share the same training/evaluation observations in each replicate.
The corrected EIF removes the historical inflated interval widths. TL improves
coverage over default boosting but under-covers with small training samples;
near-nominal coverage requires thousands, rather than hundreds, of training
observations in this experiment. Tuned boosting can have comparable coverage.
See docs/eif-regeneration.md for the results and interpretation changes.
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
    # TL uses the same default learner as the naive-default comparison.
    # Its coverage still depends on nuisance accuracy and training sample size.
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
    xlabel = "Nuisance-model training size"
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))
    # Compare observed coverage with the nominal 95% reference.
    ax = axes[0]
    sns.lineplot(data=df, x="train_size", y="coverage", hue="method",
                 marker="o", ax=ax)
    ax.axhline(0.95, ls="--", color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.02)
    ax.legend(title=None, fontsize=7)
    # Right: bias (mean estimate - truth) -- the mechanism behind the coverage
    # story: the linear bias is frozen, default GB plateaus, tuned GB shrinks to 0,
    # and TL reduces, but need not eliminate, nuisance-related bias.
    ax = axes[1]
    sns.lineplot(data=df, x="train_size", y="bias", hue="method",
                 marker="o", ax=ax, legend=False)
    ax.axhline(0.0, ls="--", color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Bias  $\overline{\hat{\Psi}} - \Psi$")
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
