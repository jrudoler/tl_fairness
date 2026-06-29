"""Experiment 3: the permutation test answers the wrong question (CMI setting).

CMI DGP (``tlfair.cmi_sim``): a confounder Z drives two binary variables; in the
data-fairness mapping the *outcome* is ``x``, the *group* is ``y``, and the
features are ``z``. The TL estimand is conditional mutual information
I(outcome; group | Z). The dependence strength is ``c``:

  * c = 0  -> outcome and group are *conditionally* independent given Z
             (I = 0, the null), but they remain *marginally* dependent because
             they share Z.
  * c > 0  -> genuine conditional dependence (I > 0).

A permutation test is a p-value, not a CI, so we evaluate calibration:
rejection rate at c=0 is the Type-I error (target ~0.05); rejection rate at
c>0 is power. Methods:

  * Global permutation -- permute the group freely; statistic = marginal MI of
    (outcome, group) ignoring Z. Tests *marginal* independence, so at c=0 it
    rejects because of the shared-Z dependence: Type-I error for the
    conditional-independence question.
  * Stratified permutation -- permute the group only within Z quantile bins;
    statistic = within-stratum (conditional) MI. The "fair" conditional analogue.
    Much closer to nominal than the global test, but coarse binning of a
    continuous Z leaves residual within-bin dependence, so its Type-I error
    still inflates as n grows -- it only conditions approximately.
  * TL Wald -- one-sided 0.05 test from the CMI EIF CI (reject if
    est - 1.645*se > 0). Calibrated/conservative away from the boundary; we
    report it honestly at c=0 given the documented boundary non-regularity.

Message: you must condition on Z. The naive global permutation does not, so it
falsely flags conditional dependence (Type-I error ~1.0); TL conditions on Z
fully and stays calibrated/conservative; the stratified permutation only
conditions approximately (coarse bins) and partly recovers calibration.

Usage (smoke):
  PYTHONPATH=. .venv/bin/python experiments/exp3_cmi_permutation.py \
      --weights 0 1 --sizes 1000 --reps 20 --n-perm 100 --n-jobs 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier

from tlfair.metrics import cmi
from tlfair.plotting import configure_matplotlib, FULL_WIDTH
# _sigmoid is scipy.special.expit re-exported from a real module, so it survives
# joblib/loky serialization (a bare ``from scipy.special import expit`` into
# __main__ does not).
from experiments.baselines import permutation_mi_test, permutation_cmi_test, _sigmoid

METHODS = ["Global permutation", "Stratified permutation", "TL Wald"]
ALPHA = 0.05
_Z2 = 1.96    # for recovering se from the two-sided 95% CMI CI
_Z1 = 1.645   # one-sided 0.05 critical value


def _draw(n, c, rng, d=3):
    z = rng.normal(size=(n, d))
    beta = np.ones(d)
    shared = c * rng.uniform(size=n)
    logits = _sigmoid(z @ beta)
    x = ((shared + rng.uniform(size=n) + logits) / (c + 2) > 0.5).astype(np.int8)
    y = ((shared + rng.uniform(size=n) + logits) / (c + 2) > 0.5).astype(np.int8)
    return x, y, z


def _one_rep(n, c, n_perm, rng, d=3, max_attempts=20):
    """One replicate; returns {method: reject(0/1)} evaluated on one draw."""
    last_err = None
    for _ in range(max_attempts):
        x, y, z = _draw(n, c, rng, d)
        h = n // 2
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            est, (lo, hi) = cmi(
                X_train=z[:h], X_test=z[h:],
                y_train=x[:h], y_test=x[h:],
                group_train=y[:h], group_test=y[h:],
                outcome=GradientBoostingClassifier(random_state=seed),
            )
            break
        except ValueError as err:  # degenerate cv=3 calibration; redraw
            last_err = err
    else:
        raise last_err

    se = (hi - lo) / (2 * _Z2)
    tl_reject = 1.0 if (est - _Z1 * se) > 0 else 0.0
    glob_p = permutation_mi_test(x, y, rng, n_perm=n_perm)
    strat_p = permutation_cmi_test(x, y, z, rng, n_perm=n_perm)
    return {
        "Global permutation": 1.0 if glob_p <= ALPHA else 0.0,
        "Stratified permutation": 1.0 if strat_p <= ALPHA else 0.0,
        "TL Wald": tl_reject,
    }


def run(weights, sizes, reps, n_perm, seed, n_jobs):
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        for c in weights:
            children = rng.spawn(reps)
            results = Parallel(n_jobs=n_jobs)(
                delayed(_one_rep)(n, c, n_perm, ch) for ch in children)
            for m in METHODS:
                rate = float(np.mean([r[m] for r in results]))
                rows.append({
                    "method": m, "c": c, "sample_size": n,
                    "reject_rate": rate,
                    "kind": "Type-I error" if c == 0 else "power",
                })
            rates = {r["method"]: r["reject_rate"] for r in rows[-len(METHODS):]}
            tag = "Type-I @ c=0" if c == 0 else f"power @ c={c}"
            print(f"  n={n} {tag}: "
                  + ", ".join(f"{m}={rates[m]:.3f}" for m in METHODS), flush=True)
    return pd.DataFrame(rows)


def plot(df, output):
    import matplotlib.pyplot as plt
    import seaborn as sns
    configure_matplotlib()
    sizes = list(df["sample_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(FULL_WIDTH, FULL_WIDTH * 0.42),
                             sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    for ax, n in zip(axes, sizes):
        sub = df[df["sample_size"] == n]
        sns.lineplot(data=sub, x="c", y="reject_rate", hue="method",
                     marker="o", ax=ax)
        ax.axhline(ALPHA, ls="--", color="grey", lw=1)
        ax.set_xlabel("Dependence strength c  (c=0 is the null)")
        ax.set_title(f"n={n}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(title=None, fontsize=7)
    axes[0].set_ylabel("Rejection rate")
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 0.5, 1, 2, 3])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 2500])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", default="experiments/out/exp3_cmi_permutation.csv")
    parser.add_argument("--figure", default="experiments/out/exp3_cmi_permutation.png")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = run(args.weights, args.sizes, args.reps, args.n_perm, args.seed, args.n_jobs)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)
    print(df.to_string(index=False), flush=True)
    plot(df, args.figure)


if __name__ == "__main__":
    main()
