"""Experiment 2: misspecification and coverage in Setting 3.

The parity panel compares TL one-step with correctly specified GLM
standardization (CLT and bootstrap intervals) and misspecified linear GLM
standardization (bootstrap). The adjusted-effect panel targets a different,
zero conditional group effect using correctly specified and linear GLMs.
Corrected TL coverage is not uniformly nominal or superior: it under-covers at
small sample sizes. Bootstrap intervals cannot remove model misspecification
bias. See docs/eif-regeneration.md for the rerun and historical comparison.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier

from tlfair.metrics import prob_parity
from tlfair.plotting import configure_matplotlib, FULL_WIDTH
from tlfair.simulations import setting3_draw, setting3_truth
from experiments.baselines import glm_ame_parity, glm_standardization_parity

# (method label, target) in display order. The parity panel isolates the two
# failure modes of a GLM-based parity estimate (standardization / g-computation):
#   * Std + CLT (correct):       correct model, but the CLT interval misses the
#                                coefficient-estimation variance -> under-covers.
#   * Std + bootstrap (correct): correct model + proper (bootstrap) inference ->
#                                covers. (So the CLT failure is a variance issue.)
#   * Std + bootstrap (linear):  misspecified model + proper inference -> still
#                                fails, now from bias. TL reduces sensitivity to nuisance error,
#                                but does not guarantee finite-sample coverage.
PARITY_METHODS = ["TL one-step (flexible)", "Std + CLT (correct)",
                  "Std + bootstrap (correct)", "Std + bootstrap (linear)"]
# GLM specs: (label, feature_set, cov_type). "correct" -> x**2, "linear" -> x.
GLM_SPECS = [
    ("GLM correct (robust)", "correct", "HC0"),
    ("GLM linear (model)", "linear", "model"),
    ("GLM linear (robust)", "linear", "HC0"),
]
GLM_METHODS = [label for label, _, _ in GLM_SPECS]


def _one_rep(n, rng, bernoulli=False, n_boot=200):
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
    # --- Panel A: probabilistic parity (all four target the parity gap) ---
    est, (lo, hi) = prob_parity(
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        group_train=gtr, group_test=gte,
        outcome=GradientBoostingClassifier(random_state=seed),
        propensity=GradientBoostingClassifier(random_state=seed + 1),
    )
    out["TL one-step (flexible)"] = (est, lo, hi)

    # GLM standardization (g-computation) of the parity gap, correct (x**2) model:
    est, (lo, hi) = glm_standardization_parity(x ** 2, g, y, ci="clt")
    out["Std + CLT (correct)"] = (est, lo, hi)
    est, (lo, hi) = glm_standardization_parity(
        x ** 2, g, y, ci="bootstrap", n_boot=n_boot, rng=rng)
    out["Std + bootstrap (correct)"] = (est, lo, hi)
    # ...and a misspecified (linear) model, also bootstrapped:
    est, (lo, hi) = glm_standardization_parity(
        x, g, y, ci="bootstrap", n_boot=n_boot, rng=rng)
    out["Std + bootstrap (linear)"] = (est, lo, hi)

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


def run(sizes, reps, seed, n_jobs, truth_n, bernoulli=False, n_boot=200):
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
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_rep)(n, c, bernoulli, n_boot) for c in children)
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
    parser.add_argument("--n-boot", type=int, default=200,
                        help="bootstrap resamples for the GLM-standardization CI.")
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
             bernoulli=not args.deterministic, n_boot=args.n_boot)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)
    print(df.to_string(index=False), flush=True)
    plot(df, args.figure)


if __name__ == "__main__":
    main()
