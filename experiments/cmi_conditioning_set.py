"""Experiment 4: CMI estimates depend on the conditioning set.

The population CMI ``I(Y;G|X)`` is zero iff Y and G are conditionally independent.
This experiment fixes a confounded data-generating process and sweeps which
features are conditioned on. Estimates and empirical Wald rejection rates are descriptive
outputs; the Wald rule is not a validated conditional-independence test.

DGP (high-dimensional, random-but-seeded coefficients). ``p`` independent
standard-normal features split into three disjoint column groups:

  * Confounders  Z = X[:, :s_conf]                  -- drive BOTH G and Y
  * Outcome pred X_O = X[:, s_conf:s_conf+s_out]     -- drive Y only
  * Noise        X_N = X[:, s_conf+s_out:]           -- drive nothing

    G ~ Bernoulli(sigmoid(Z @ alpha))                        # depends only on Z
    Y ~ Bernoulli(sigmoid(Z @ beta_conf + X_O @ beta_out))   # drawn ⟂ G given X

Because G and Y share only Z and are drawn independently given X, ``Y ⊥ G | Z``
holds *exactly*, so conditioning on any feature set that CONTAINS Z gives true
CMI = 0. Dropping the confounders -- even while
adjusting for every other (predictive) feature -- leaves the confounding through
Z unexplained, giving positive CMI in this design. Predictive features need not
suffice to remove conditional dependence.
Marginally, Y and G remain dependent through the shared Z (reported as the
empirical MI(Y;G)), so there IS marginal bias that the right conditioning explains
away.

The swept axis is ``k`` = how many of the ``s_conf`` confounders are included in
the conditioning set (the outcome predictors + noise are ALWAYS included, so the
contrast is purely "did you adjust for the confounders?"). The diagnostic rule
mirrors exp3: reject if est - 1.645*se > 0. Its nominal 5% cutoff is not justified
at the CMI null, where the EIF vanishes. Low rejection rates can coexist with poor
interval coverage and do not establish independence. Legacy CSV fields
``reject_rate`` and ``kind`` retain their names for compatibility.

The DGP is defined inline here (like exp3._draw); promote it to
``tlfair/simulations.py`` (e.g. ``confounded_feature_draw``) once it settles.

Usage (smoke):
  PYTHONPATH=. .venv/bin/python experiments/cmi_conditioning_set.py \
      --p 20 --n-confounders 3 --n-outcome 5 --sizes 1000 --reps 20 --n-jobs 4
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tlfair.metrics import cmi
from tlfair.plotting import configure_matplotlib, FULL_WIDTH
# _sigmoid is scipy.special.expit re-exported from a real module, so it survives
# joblib/loky serialization (a bare import into __main__ does not); _binary_mi
# backs the marginal-dependence reference reported to stdout.
from experiments.baselines import _sigmoid, _binary_mi

ALPHA = 0.05
_Z2 = 1.96    # two-sided 95% -> recover se from the CMI CI half-width
_Z1 = 1.645   # one-sided 0.05 critical value


LOG = logging.getLogger(__name__)

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
    all non-confounders only (highest rejection rate) and k=n_conf conditions on the full
    feature set (true CMI = 0).
    """
    return list(range(k)) + list(range(n_conf, p))


def _one_rep(n, k, p, n_conf, n_out, coefs, rng, max_attempts=20):
    """Paired methods share the DGP, split, seed and early-stopped joint fit."""
    from tlfair.cmi_estimation import (fit_joint_probabilities, paired_cmi, replicate_records,
                                      make_cmi_learner, joint_fit_metadata)
    from tlfair.metrics import _encode_joint
    cols = _conditioning_cols(k, p, n_conf, n_out)
    h = n // 2
    last_err = None
    for _ in range(max_attempts):
        x, g, y = _draw(2*h, p, n_conf, n_out, coefs, rng)
        xc = x[:, cols]
        seed = int(rng.integers(0, 2**31-1))
        try:
            if np.bincount(_encode_joint(g[:h],y[:h]), minlength=4).min() < 3:
                raise ValueError('Insufficient training observations for stratified joint fitting')
            learner = make_cmi_learner(seed)
            q = fit_joint_probabilities(xc[:h], xc[h:], y[:h], g[:h],
                                       learner)
            result = paired_cmi(q, _encode_joint(g[h:],y[h:]))
            result['nuisance_fit'] = joint_fit_metadata(learner)
            rows = replicate_records(result)
            for row in rows:
                row['lower_one_sided'] = row['estimate']-_Z1*row['se']
                row['rejects_zero'] = float(row['lower_one_sided'] > 0)
            return rows
        except ValueError as err:
            last_err = err
    raise last_err


def summarize_records(records: pd.DataFrame, n_conf: int) -> pd.DataFrame:
    records = records.copy()
    records['width'] = records.ci_high-records.ci_low
    records['covered'] = ((records.ci_low <= 0)&(records.ci_high >= 0)).astype(float)
    result = records.groupby(['sample_size','n_confounders_included','estimator']).agg(
        reject_rate=('rejects_zero','mean'), mean_estimate=('estimate','mean'),
        mean_ci_width=('width','mean'), coverage_at_null=('covered','mean'),
        reps=('replicate','size')).reset_index()
    result.loc[result.n_confounders_included != n_conf,'coverage_at_null'] = np.nan
    return result


def _marginal_mi(p, n_conf, n_out, coefs, seed, n=200_000):
    """Empirical marginal MI(Y;G) on a large draw -- establishes marginal bias."""
    rng = np.random.default_rng(seed)
    _, g, y = _draw(n, p, n_conf, n_out, coefs, rng)
    return _binary_mi(y, g)


def run(p, n_conf, n_out, sizes, reps, signal, coef_seed, seed, n_jobs, records_path=None):
    coefs = _make_coefs(n_conf, n_out, signal, coef_seed)
    marg_mi = _marginal_mi(p, n_conf, n_out, coefs, seed)
    LOG.info('Marginal MI(Y;G) = %.5f nats', marg_mi)

    from tlfair.cmi_estimation import require_convergence
    records = []
    for n in sizes:
        for k in range(n_conf+1):
            children = np.random.default_rng(np.random.SeedSequence([seed,n,k])).spawn(reps)
            batches = Parallel(n_jobs=n_jobs)(
                delayed(_one_rep)(n,k,p,n_conf,n_out,coefs,ch) for ch in children)
            records.extend(dict(row,sample_size=n,n_confounders_included=k,replicate=i)
                           for i,batch in enumerate(batches) for row in batch)
            if records_path is not None:
                records_path = Path(records_path)
                records_path.parent.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(records).to_csv(records_path,index=False)
            LOG.info('Conditioning-set cell completed: n=%d included=%d/%d', n,k,n_conf)
    frame = pd.DataFrame(records)
    require_convergence(frame)
    result = summarize_records(frame,n_conf)
    result['coef_seed'] = coef_seed
    result['n_features'] = result.n_confounders_included.map(lambda k:len(_conditioning_cols(k,p,n_conf,n_out)))
    result['sample_size'] = result.sample_size.astype(int)
    result['kind'] = np.where(result.n_confounders_included == n_conf,'null diagnostic','partial conditioning')
    result['marginal_mi'] = marg_mi
    return result


def plot(df: pd.DataFrame, output: str | Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    if 'estimator' in df:
        df = df[df.estimator == 'tmle']
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))
    # Left: mean CMI estimate and mean endpoints of two-sided 95% intervals.
    ax = axes[0]
    for n, sub in df.groupby("sample_size"):
        sub = sub.sort_values("n_confounders_included")
        x = sub["n_confounders_included"].to_numpy()
        m = sub["mean_estimate"].to_numpy()
        half_width = sub["mean_ci_width"].to_numpy() / 2
        line, = ax.plot(x, m, marker="o", label=f"n={n}")
        ax.fill_between(x, m - half_width, m + half_width,
                        color=line.get_color(), alpha=0.15)
    ax.axhline(0.0, ls="--", color="grey", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\hat{I}(Y;G\mid X)$")
    ax.legend(title=None, fontsize=7)
    # Right: empirical rejection rate vs #confounders included.
    ax = axes[1]
    sns.lineplot(data=df, x="n_confounders_included", y="reject_rate",
                 hue="sample_size", marker="o", ax=ax, palette="tab10")
    ax.axhline(ALPHA, ls="--", color="grey", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("Wald rejection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="n", fontsize=7)
    fig.supxlabel("# confounders in conditioning set", fontsize=plt.rcParams["axes.labelsize"])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
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
    parser.add_argument("--output", default="experiments/out/cmi_conditioning_set.csv")
    parser.add_argument("--figure", default="experiments/out/cmi_conditioning_set.png")
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

    # Report the observed rates without treating low null rejection as a pass.
    big = df[df["sample_size"] == max(args.sizes)]
    null_rate = big[big["n_confounders_included"] == args.n_confounders]["reject_rate"].iloc[0]
    rejection_rate = big[big["n_confounders_included"] == 0]["reject_rate"].iloc[0]
    print(f"\nObserved @ n={max(args.sizes)}: with-Z rejection rate={null_rate:.3f}, "
          f"without-Z rejection rate={rejection_rate:.3f}; null calibration is not established.",
          flush=True)


if __name__ == "__main__":
    main()
