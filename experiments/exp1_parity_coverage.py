"""Experiment 1: coverage head-to-head for probabilistic parity (Setting 1).

Same estimand, two methods:
  * TL one-step (``tlfair.metrics.prob_parity``) -- EIF-based Wald CI.
  * Naive fixed-model + CLT -- one model's per-group prediction means with a
    two-sample-mean CI (the "model fairness" straw man).

Factor: the feature set handed to the (logistic) learners.
  * "correct"  -- includes the true product terms X1*X4, X2*X3 (well specified).
  * "linear"   -- raw features only (mildly misspecified outcome model).

This is the regime *most favourable to the straw man*: a low-dimensional,
(near-)well-specified parametric model. Here the naive CLT is approximately
calibrated -- when the logistic model converges fast, the uncertainty it ignores
(from estimating P(Y|X)) is negligible, so its narrower interval (cf. the
variance gap of Figure 3) still roughly covers. TL is valid throughout but
*conservative* (it never under-covers). The naive interval starts to slip under
the mild "linear" misspecification (~0.92), foreshadowing Experiment 2, where
flexible/misspecified models make the naive CLT under-cover badly while TL stays
calibrated. Takeaway: the naive "model fairness" CI and the TL "data fairness" CI
nearly coincide only when the model is easy to estimate; they diverge otherwise.

Usage (smoke):
  PYTHONPATH=. .venv/bin/python experiments/exp1_parity_coverage.py \
      --sizes 250 500 --reps 20 --n-jobs 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from tlfair.metrics import prob_parity
from tlfair.plotting import configure_matplotlib, FULL_WIDTH
from tlfair.simulations import _setting1_draw, parity_ground_truth
from experiments.baselines import naive_fixed_model_parity

FEATURE_SETS = ["correct", "linear"]
METHODS = ["TL one-step", "Naive fixed + CLT"]


def _design(xg, feature_set):
    """Build the feature matrix handed to the learners for a given spec."""
    if feature_set == "correct":
        x14 = (xg[:, 1] * xg[:, 4]).reshape(-1, 1)
        x23 = (xg[:, 2] * xg[:, 3]).reshape(-1, 1)
        return np.hstack([xg, x14, x23])
    return xg  # "linear": raw features only -> misspecified outcome model


def _one_rep(n, feature_set, rng, bernoulli=False):
    """One Setting-1 replicate; returns {method: (est, lo, hi)}."""
    h = n // 2
    xg, g, y, _ = _setting1_draw(2 * h, rng, bernoulli=bernoulli)
    X = _design(xg, feature_set)
    Xtr, Xte = X[:h], X[h:]
    ytr, yte = y[:h], y[h:]
    gtr, gte = g[:h], g[h:]

    out = {}
    est, (lo, hi) = prob_parity(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        group_train=gtr, group_test=gte,
        outcome=LogisticRegression(max_iter=500),
        propensity=LogisticRegression(max_iter=500),
    )
    out["TL one-step"] = (est, lo, hi)

    est, (lo, hi) = naive_fixed_model_parity(
        LogisticRegression(max_iter=500), Xtr, ytr, Xte, gte)
    out["Naive fixed + CLT"] = (est, lo, hi)
    return out


def _aggregate(results, truth, feature_set, n):
    rows = []
    for method in METHODS:
        ests = np.array([r[method][0] for r in results])
        los = np.array([r[method][1] for r in results])
        his = np.array([r[method][2] for r in results])
        covered = (los <= truth) & (truth <= his)
        rows.append({
            "feature_set": feature_set,
            "method": method,
            "sample_size": n,
            "coverage": float(np.mean(covered)),
            "bias": float(np.mean(ests) - truth),
            "mean_ci_width": float(np.mean(his - los)),
            "mean_estimate": float(np.mean(ests)),
            "truth": truth,
        })
    return rows


def run(sizes, reps, seed, n_jobs, truth_n, bernoulli=False):
    rng = np.random.default_rng(seed)
    truth = parity_ground_truth(n=truth_n, rng=rng)[1]  # probabilistic parity (unchanged by noise)
    print(f"Setting-1 probabilistic-parity truth = {truth:.5f}"
          f"  [outcome={'Bernoulli' if bernoulli else 'deterministic'}]", flush=True)
    rows = []
    for feature_set in FEATURE_SETS:
        for n in sizes:
            children = rng.spawn(reps)
            results = Parallel(n_jobs=n_jobs)(
                delayed(_one_rep)(n, feature_set, c, bernoulli) for c in children)
            rows.extend(_aggregate(results, truth, feature_set, n))
            cov = {r["method"]: r["coverage"] for r in rows[-len(METHODS):]}
            print(f"  [{feature_set}] n={n}: "
                  + ", ".join(f"{m}={cov[m]:.3f}" for m in METHODS), flush=True)
    return pd.DataFrame(rows)


def plot(df, output):
    import matplotlib.pyplot as plt
    import seaborn as sns
    configure_matplotlib()
    fsets = list(df["feature_set"].unique())
    fig, axes = plt.subplots(1, len(fsets), figsize=(FULL_WIDTH, FULL_WIDTH * 0.42),
                             sharey=True)
    if len(fsets) == 1:
        axes = [axes]
    for ax, fs in zip(axes, fsets):
        sub = df[df["feature_set"] == fs]
        sns.lineplot(data=sub, x="sample_size", y="coverage", hue="method",
                     marker="o", ax=ax)
        ax.axhline(0.95, ls="--", color="grey", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Sample size")
        ax.set_title(f"{fs} features")
        ax.set_ylim(0, 1.02)
        ax.legend(title=None, fontsize=8)
    axes[0].set_ylabel("95% CI coverage")
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[250, 500, 1000, 2500, 5000])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--truth-n", type=int, default=2_000_000)
    parser.add_argument("--deterministic", action="store_true",
                        help="use the deterministic Bayes-decision outcome instead of "
                             "the default Bernoulli draw (Bernoulli is canonical; it "
                             "avoids perfect separation -- see experiments/audit_glm.py).")
    parser.add_argument("--output", default="experiments/out/exp1_parity_coverage.csv")
    parser.add_argument("--figure", default="experiments/out/exp1_parity_coverage.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = run(args.sizes, args.reps, args.seed, args.n_jobs, args.truth_n,
             bernoulli=not args.deterministic)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)
    print(df.to_string(index=False), flush=True)
    plot(df, args.figure)


if __name__ == "__main__":
    main()
