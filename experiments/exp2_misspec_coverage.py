"""Experiment 2: misspecification breaks the naive baselines (Setting 3).

Setting 3: Y and G are both logistic in the *squared* features, so a linear
logistic model is misspecified for both nuisances (paper Figure 2). Two targets,
reported in two panels because the methods answer different questions:

Panel A -- probabilistic parity Ψ = E[D(X)|G=1] - E[D(X)|G=0] (the data-fairness
target, truth = ``setting3_truth``):
  * TL one-step, flexible nuisances (gradient boosting): double-robust, stays
    calibrated under misspecification.
  * Naive fixed-model + CLT, flexible model: ~unbiased point estimate but the
    CLT interval ignores P(Y|X) uncertainty -> under-covers.
  * Naive fixed-model + CLT, linear model: misspecified -> biased *and* the
    too-narrow interval under-covers badly.

Panel B -- GLM "adjusted group effect" (the average marginal effect of G with X
held fixed; what regression-coefficient practice reports). Here G affects Y only
through X, so the true adjusted effect is structurally 0 -- and note this differs
from the real parity gap (printed for contrast), so the GLM answers a *different*
question. To make the failure attributable to BIAS rather than an optimistic
variance, the GLM CIs use the Huber-White sandwich (HC0) SE; a model-based
variant is included for the linear fit to show the sandwich barely moves the AME
SE, so the under-coverage is driven by bias in the estimate:
  * GLM correct (squared features), robust SE: ~0, covers 0. (Caveat: the
    noiseless DGP makes the correct-feature fit perfectly separable -- a
    degenerate MLE; see experiments/audit_glm.py.)
  * GLM linear, model vs robust SE: misspecified -> AME biased away from 0; the
    CI (either SE) is tight around the wrong value -> under-covers. The sandwich
    rules out "the SE was just too small" as the explanation.

Usage (smoke):
  PYTHONPATH=. .venv/bin/python experiments/exp2_misspec_coverage.py \
      --sizes 250 500 --reps 20 --n-jobs 4
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
from experiments.baselines import naive_fixed_model_parity, glm_ame_parity

# (method label, target) in display order.
PARITY_METHODS = ["TL one-step (flexible)", "Naive+CLT (flexible)", "Naive+CLT (linear)"]
# GLM specs: (label, feature_set, cov_type). "correct" -> x**2, "linear" -> x.
GLM_SPECS = [
    ("GLM correct (robust)", "correct", "HC0"),
    ("GLM linear (model)", "linear", "model"),
    ("GLM linear (robust)", "linear", "HC0"),
]
GLM_METHODS = [label for label, _, _ in GLM_SPECS]


def _one_rep(n, rng, bernoulli=False):
    """One Setting-3 replicate; returns {method: (est, lo, hi)}."""
    h = n // 2
    # Match the manuscript: both Y and G are Bernoulli (the latter also restores
    # positivity). Tie the group draw to the same flag as the outcome.
    x, g, y, _ = setting3_draw(2 * h, rng, bernoulli=bernoulli, bernoulli_group=bernoulli)
    Xtr, Xte = x[:h], x[h:]
    ytr, yte = y[:h], y[h:]
    gtr, gte = g[:h], g[h:]
    seed = int(rng.integers(0, 2**31 - 1))  # reproducible GB fits

    out = {}
    # --- Panel A: probabilistic parity ---
    est, (lo, hi) = prob_parity(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        group_train=gtr, group_test=gte,
        outcome=GradientBoostingClassifier(random_state=seed),
        propensity=GradientBoostingClassifier(random_state=seed + 1),
    )
    out["TL one-step (flexible)"] = (est, lo, hi)

    est, (lo, hi) = naive_fixed_model_parity(
        GradientBoostingClassifier(random_state=seed), Xtr, ytr, Xte, gte)
    out["Naive+CLT (flexible)"] = (est, lo, hi)

    est, (lo, hi) = naive_fixed_model_parity(
        LogisticRegression(max_iter=500), Xtr, ytr, Xte, gte)
    out["Naive+CLT (linear)"] = (est, lo, hi)

    # --- Panel B: GLM adjusted group effect (uses full sample) ---
    glm_feats = {"correct": x ** 2, "linear": x}
    for label, fs, cov_type in GLM_SPECS:
        est, (lo, hi) = glm_ame_parity(glm_feats[fs], g, y, cov_type=cov_type)
        out[label] = (est, lo, hi)
    return out


def _aggregate(results, method, target, truth, n):
    ests = np.array([r[method][0] for r in results])
    los = np.array([r[method][1] for r in results])
    his = np.array([r[method][2] for r in results])
    covered = (los <= truth) & (truth <= his)
    return {
        "target": target,
        "method": method,
        "sample_size": n,
        "coverage": float(np.mean(covered)),
        "bias": float(np.mean(ests) - truth),
        "mean_ci_width": float(np.mean(his - los)),
        "mean_estimate": float(np.mean(ests)),
        "truth": truth,
    }


def run(sizes, reps, seed, n_jobs, truth_n, bernoulli=False):
    rng = np.random.default_rng(seed)
    parity_truth = setting3_truth(truth_n, rng, bernoulli_group=bernoulli)  # group law must match data
    ace_truth = 0.0  # G affects Y only through X -> adjusted effect is exactly 0
    print(f"Setting-3 probabilistic-parity truth = {parity_truth:.5f}"
          f"  [outcome={'Bernoulli' if bernoulli else 'deterministic'}]", flush=True)
    print(f"Setting-3 adjusted-effect truth (GLM target) = {ace_truth:.5f} "
          f"(differs from parity -> GLM answers a different question)", flush=True)

    rows = []
    for n in sizes:
        children = rng.spawn(reps)
        results = Parallel(n_jobs=n_jobs)(delayed(_one_rep)(n, c, bernoulli) for c in children)
        for m in PARITY_METHODS:
            rows.append(_aggregate(results, m, "parity", parity_truth, n))
        for m in GLM_METHODS:
            rows.append(_aggregate(results, m, "adjusted effect (truth 0)", ace_truth, n))
        cov = {r["method"]: r["coverage"] for r in rows[-(len(PARITY_METHODS) + len(GLM_METHODS)):]}
        print(f"  n={n}: " + ", ".join(f"{m}={cov[m]:.3f}"
              for m in PARITY_METHODS + GLM_METHODS), flush=True)
    return pd.DataFrame(rows)


def plot(df, output):
    import matplotlib.pyplot as plt
    import seaborn as sns
    configure_matplotlib()
    targets = list(df["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(FULL_WIDTH, FULL_WIDTH * 0.42),
                             sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub = df[df["target"] == tgt]
        sns.lineplot(data=sub, x="sample_size", y="coverage", hue="method",
                     marker="o", ax=ax)
        ax.axhline(0.95, ls="--", color="grey", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Sample size")
        ax.set_title(tgt)
        ax.set_ylim(0, 1.02)
        ax.legend(title=None, fontsize=7)
    axes[0].set_ylabel("95% CI coverage")
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000, 2500])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--truth-n", type=int, default=10_000_000)
    parser.add_argument("--deterministic", action="store_true",
                        help="use the deterministic Bayes-decision outcome instead of "
                             "the default Bernoulli draw (Bernoulli is canonical; it "
                             "makes the well-specified GLM non-separable, hence a fair "
                             "baseline -- see experiments/audit_glm.py).")
    parser.add_argument("--output", default="experiments/out/exp2_misspec_coverage.csv")
    parser.add_argument("--figure", default="experiments/out/exp2_misspec_coverage.png")
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
