"""Experiment 4: the data-fairness verdict depends on the conditioning set.

Data fairness is the conditional-independence property ``Y ⊥ G | X``, certified
via the CMI estimand ``I(Y;G|X)`` (=0 iff conditionally independent;
``tlfair.metrics.cmi``). Every other simulation fixes the feature set; this one
fixes a *confounded* data-generating process and sweeps **which features are
conditioned on**, showing the verdict flips with the conditioning set.

DGP (high-dimensional, random-but-seeded coefficients). ``p`` independent
standard-normal features split into three disjoint column groups:

  * Confounders  Z = X[:, :s_conf]                  -- drive BOTH G and Y
  * Outcome pred X_O = X[:, s_conf:s_conf+s_out]     -- drive Y only
  * Noise        X_N = X[:, s_conf+s_out:]           -- drive nothing

    G ~ Bernoulli(sigmoid(Z @ alpha))                        # depends only on Z
    Y ~ Bernoulli(sigmoid(Z @ beta_conf + X_O @ beta_out))   # drawn ⟂ G given X

Because G and Y share only Z and are drawn independently given X, ``Y ⊥ G | Z``
holds *exactly*, so conditioning on any feature set that CONTAINS Z gives true
CMI = 0 (the data-fairness test passes). Dropping the confounders -- even while
adjusting for every other (predictive) feature -- leaves the confounding through
Z unexplained, so CMI > 0 and the test flags the setup as unfair: predictive
power is not fairness sufficiency, and the minimal sufficient adjustment set is Z.
Marginally, Y and G remain dependent through the shared Z (reported as the
empirical MI(Y;G)), so there IS marginal bias that the right conditioning explains
away.

The swept axis is ``k`` = how many of the ``s_conf`` confounders are included in
the conditioning set (the outcome predictors + noise are ALWAYS included, so the
contrast is purely "did you adjust for the confounders?"). The data-fairness
verdict mirrors exp3's one-sided TL Wald test: flag unfair if est - 1.645*se > 0.
At k = s_conf the rejection rate is the Type-I error (should be low / calibrated
-> PASS); at k < s_conf it is power (-> FLAG).

The DGP is defined inline here (like exp3._draw); promote it to
``tlfair/simulations.py`` (e.g. ``confounded_feature_draw``) once it settles.

Usage (smoke):
  PYTHONPATH=. .venv/bin/python experiments/exp4_feature_selection.py \
      --p 20 --n-confounders 3 --n-outcome 5 --sizes 1000 --reps 20 --n-jobs 4
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
# joblib/loky serialization (a bare import into __main__ does not); _binary_mi
# backs the marginal-dependence reference reported to stdout.
from experiments.baselines import _sigmoid, _binary_mi

ALPHA = 0.05
_Z2 = 1.96    # two-sided 95% -> recover se from the CMI CI half-width
_Z1 = 1.645   # one-sided 0.05 critical value


def _make_coefs(n_conf, n_out, signal, coef_seed):
    """Draw the (fixed, seeded) DGP coefficients once.

    Coefficients are genuine random variables -- reproducibility comes from the
    seed, not from hand-crafting. The confounder effects on G (``alpha``) and on Y
    (``beta_conf``) are drawn with a **shared per-confounder sign** so each
    confounder pushes G and Y in the same direction: this is what makes the shared
    Z channel actually couple G and Y (independent random directions are nearly
    orthogonal in R^{n_conf}, giving negligible marginal dependence). Each
    confounder then contributes to the coupling, so dropping any of them from the
    conditioning set leaves detectable residual dependence. Magnitudes are
    ``signal * Uniform(0.5, 1.5)``; outcome-only coefficients are unconstrained
    (they predict Y but must not couple with G).
    """
    crng = np.random.default_rng(coef_seed)
    signs = crng.choice([-1.0, 1.0], size=n_conf)
    alpha = signs * signal * crng.uniform(0.5, 1.5, size=n_conf)       # G on Z
    beta_conf = signs * signal * crng.uniform(0.5, 1.5, size=n_conf)   # Y on Z (aligned)
    beta_out = signal * crng.uniform(0.5, 1.5, size=n_out) * crng.choice(
        [-1.0, 1.0], size=n_out)                                        # Y on X_O
    return alpha, beta_conf, beta_out


def _draw(n, p, n_conf, n_out, coefs, rng):
    """One confounded draw. Returns (X (n,p), g (n,), y (n,)).

    Z = X[:, :n_conf] drives both G and Y; X_O = X[:, n_conf:n_conf+n_out] drives
    Y only; the rest is noise. G and Y are drawn independently given X, so
    Y ⟂ G | Z exactly.
    """
    alpha, beta_conf, beta_out = coefs
    x = rng.normal(size=(n, p))
    z = x[:, :n_conf]
    x_o = x[:, n_conf:n_conf + n_out]
    g = (rng.uniform(size=n) < _sigmoid(z @ alpha)).astype(np.int8)
    y_logit = z @ beta_conf + (x_o @ beta_out if n_out else 0.0)
    y = (rng.uniform(size=n) < _sigmoid(y_logit)).astype(np.int8)
    return x, g, y


def _conditioning_cols(k, p, n_conf, n_out):
    """Columns for including ``k`` of the ``n_conf`` confounders.

    Outcome predictors + noise (everything from column ``n_conf`` on) are always
    included; ``k`` selects a prefix of the confounder block. So k=0 conditions on
    all non-confounders only (strongest flag) and k=n_conf conditions on the full
    feature set (pass).
    """
    return list(range(k)) + list(range(n_conf, p))


def _one_rep(n, k, p, n_conf, n_out, coefs, rng, max_attempts=20):
    """One replicate at conditioning level ``k``; returns (est, se, reject)."""
    cols = _conditioning_cols(k, p, n_conf, n_out)
    h = n // 2
    last_err = None
    for _ in range(max_attempts):
        x, g, y = _draw(2 * h, p, n_conf, n_out, coefs, rng)
        xc = x[:, cols]
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            est, (lo, hi) = cmi(
                X_train=xc[:h], X_test=xc[h:],
                y_train=y[:h], y_test=y[h:],          # outcome
                group_train=g[:h], group_test=g[h:],  # group
                outcome=GradientBoostingClassifier(random_state=seed),
            )
            break
        except ValueError as err:  # degenerate cv=3 calibration; redraw
            last_err = err
    else:
        raise last_err
    se = (hi - lo) / (2 * _Z2)
    reject = 1.0 if (est - _Z1 * se) > 0 else 0.0
    return est, se, reject


def _marginal_mi(p, n_conf, n_out, coefs, seed, n=200_000):
    """Empirical marginal MI(Y;G) on a large draw -- establishes marginal bias."""
    rng = np.random.default_rng(seed)
    _, g, y = _draw(n, p, n_conf, n_out, coefs, rng)
    return _binary_mi(y, g)


def run(p, n_conf, n_out, sizes, reps, signal, coef_seed, seed, n_jobs):
    coefs = _make_coefs(n_conf, n_out, signal, coef_seed)
    marg_mi = _marginal_mi(p, n_conf, n_out, coefs, seed)
    print(f"Marginal MI(Y;G) = {marg_mi:.5f} nats  (>0 -> marginally biased; "
          f"conditioning on Z should drive CMI -> 0)", flush=True)

    rng = np.random.default_rng(seed)
    ks = list(range(n_conf + 1))
    rows = []
    for n in sizes:
        for k in ks:
            children = rng.spawn(reps)
            out = Parallel(n_jobs=n_jobs)(
                delayed(_one_rep)(n, k, p, n_conf, n_out, coefs, ch) for ch in children)
            ests = np.array([o[0] for o in out])
            ses = np.array([o[1] for o in out])
            rejects = np.array([o[2] for o in out])
            full = (k == n_conf)  # true CMI = 0 only when all confounders are included
            rows.append({
                "coef_seed": coef_seed,
                "n_features": len(_conditioning_cols(k, p, n_conf, n_out)),
                "n_confounders_included": k,
                "sample_size": n,
                "reps": reps,
                "reject_rate": float(np.mean(rejects)),
                "kind": "Type-I error" if full else "power",
                "mean_estimate": float(np.mean(ests)),
                "mean_ci_width": float(np.mean(2 * _Z2 * ses)),
                # coverage of the truth=0 null; only meaningful where truth is 0.
                "coverage_at_null": float(np.mean((ests - _Z2 * ses <= 0)
                                                  & (ests + _Z2 * ses >= 0))) if full else np.nan,
            })
            r = rows[-1]
            tag = "PASS (k=all Z)" if full else "flag"
            print(f"  n={n} k={k}/{n_conf} [{tag}]: reject={r['reject_rate']:.3f} "
                  f"CMI_hat={r['mean_estimate']:.4f} width={r['mean_ci_width']:.4f}",
                  flush=True)
    return pd.DataFrame(rows)


def plot(df, output):
    import matplotlib.pyplot as plt
    import seaborn as sns
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))
    # Left: mean CMI estimate vs #confounders included, with mean-CI-width band.
    ax = axes[0]
    for n, sub in df.groupby("sample_size"):
        sub = sub.sort_values("n_confounders_included")
        x = sub["n_confounders_included"].to_numpy()
        m = sub["mean_estimate"].to_numpy()
        hw = sub["mean_ci_width"].to_numpy() / 2
        line, = ax.plot(x, m, marker="o", label=f"n={n}")
        ax.fill_between(x, m - hw, m + hw, color=line.get_color(), alpha=0.15)
    ax.axhline(0.0, ls="--", color="grey", lw=1)
    ax.set_xlabel("# confounders in conditioning set")
    ax.set_ylabel(r"$\hat{I}(Y;G\mid X)$")
    ax.set_title("CMI estimate (truth = 0 with all Z)")
    ax.legend(title=None, fontsize=7)
    # Right: flag (reject) rate vs #confounders included.
    ax = axes[1]
    sns.lineplot(data=df, x="n_confounders_included", y="reject_rate",
                 hue="sample_size", marker="o", ax=ax, palette="tab10")
    ax.axhline(ALPHA, ls="--", color="grey", lw=1)
    ax.set_xlabel("# confounders in conditioning set")
    ax.set_ylabel("Flag rate (reject fairness)")
    ax.set_title("Data-fairness verdict")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="n", fontsize=7)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=30, help="total number of features")
    parser.add_argument("--n-confounders", type=int, default=5,
                        help="confounders driving both G and Y (the needed adjustment set)")
    parser.add_argument("--n-outcome", type=int, default=8,
                        help="outcome-only predictors (drive Y, independent of G)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 2500])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--signal", type=float, default=1.0,
                        help="coefficient scale (larger -> stronger dependence)")
    parser.add_argument("--coef-seed", type=int, default=0,
                        help="seed for the (fixed) DGP coefficients")
    parser.add_argument("--seed", type=int, default=123, help="seed for the data draws")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", default="experiments/out/exp4_feature_selection.csv")
    parser.add_argument("--figure", default="experiments/out/exp4_feature_selection.png")
    args = parser.parse_args()

    if args.n_confounders + args.n_outcome > args.p:
        parser.error("--n-confounders + --n-outcome must not exceed --p")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = run(args.p, args.n_confounders, args.n_outcome, args.sizes, args.reps,
             args.signal, args.coef_seed, args.seed, args.n_jobs)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}", flush=True)
    print(df.to_string(index=False), flush=True)
    plot(df, args.figure)

    # Sanity check the headline flip at the largest sample size: full-Z should
    # pass (low flag rate) and no-Z should flag (high flag rate). A warning, not a
    # hard failure -- tune --signal if the marginal dependence is too weak/strong.
    big = df[df["sample_size"] == max(args.sizes)]
    pass_rate = big[big["n_confounders_included"] == args.n_confounders]["reject_rate"].iloc[0]
    flag_rate = big[big["n_confounders_included"] == 0]["reject_rate"].iloc[0]
    print(f"\nHeadline @ n={max(args.sizes)}: with-Z flag rate={pass_rate:.3f} "
          f"(want low), without-Z flag rate={flag_rate:.3f} (want high)", flush=True)
    if not (pass_rate <= 0.15 and flag_rate >= 0.85):
        print("WARNING: the with-Z/without-Z flip is weak at this scale; "
              "consider adjusting --signal, --reps, or --sizes.", flush=True)


if __name__ == "__main__":
    main()
