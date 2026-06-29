"""Audit: cross-check the hand-rolled GLM (experiments/baselines.py) against
statsmodels, and exercise the perfect-separation pathology of the noiseless DGP.

Two things are validated:
  1. On well-behaved (non-separable) data, the hand-rolled coefficients, the
     model-based covariance (D'WD)^{-1}, the HC0 sandwich, and the discrete
     average marginal effect (AME) of G all match statsmodels Logit to numerical
     precision.
  2. On the "well-specified" feature set, the deterministic-threshold DGP is
     perfectly separable, so statsmodels' MLE diverges / raises
     PerfectSeparationError -- confirming that the well-specified GLM "fit" is a
     degenerate pseudo-MLE, not a textbook fit.

Run:
  PYTHONPATH=. .venv/bin/python experiments/audit_glm.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from tlfair.simulations import setting3_draw
from experiments.baselines import glm_coef_cov, glm_ame_parity


def _discrete_ame_and_se(beta, exog, cov):
    """Discrete AME of the last column (the group) + delta-method SE, given a
    coefficient vector and covariance -- the same estimand the hand-rolled code
    computes, but driven by arbitrary (beta, cov) so we can feed statsmodels'."""
    D1 = exog.copy(); D1[:, -1] = 1.0
    D0 = exog.copy(); D0[:, -1] = 0.0
    p1 = expit(D1 @ beta)
    p0 = expit(D0 @ beta)
    ame = float(np.mean(p1 - p0))
    grad = np.mean((p1 * (1 - p1))[:, None] * D1 - (p0 * (1 - p0))[:, None] * D0, axis=0)
    se = float(np.sqrt(grad @ cov @ grad))
    return ame, se


def cross_check_nonseparable():
    print("=" * 78)
    print("1. CROSS-CHECK on non-separable (misspecified, linear-feature) data")
    print("   Setting 3, GLM design = [1, x, G]  (true boundary is in x^2)")
    print("=" * 78)
    rng = np.random.default_rng(11)
    x, g, y, _ = setting3_draw(4000, rng)
    feats = np.hstack([x, g.reshape(-1, 1)])           # [x, G]
    exog = sm.add_constant(feats)                       # [1, x, G]  (const first)

    # --- hand-rolled core ---
    beta_hr, D, p_hr, model_cov_hr, robust_cov_hr = glm_coef_cov(x, g, y)

    # --- statsmodels: model-based and HC0 ---
    res_m = sm.Logit(y, exog).fit(disp=0)
    res_r = sm.Logit(y, exog).fit(cov_type="HC0", disp=0)

    print(f"\n{'param':>8} {'hand-rolled beta':>17} {'statsmodels beta':>17} {'abs diff':>10}")
    for j in range(len(beta_hr)):
        print(f"{j:>8} {beta_hr[j]:>17.6f} {res_m.params[j]:>17.6f} "
              f"{abs(beta_hr[j]-res_m.params[j]):>10.2e}")

    se_m_hr = np.sqrt(np.diag(model_cov_hr))
    se_r_hr = np.sqrt(np.diag(robust_cov_hr))
    print(f"\n{'param':>8} {'HR model SE':>12} {'sm model SE':>12} "
          f"{'HR HC0 SE':>12} {'sm HC0 SE':>12}")
    for j in range(len(beta_hr)):
        print(f"{j:>8} {se_m_hr[j]:>12.5f} {res_m.bse[j]:>12.5f} "
              f"{se_r_hr[j]:>12.5f} {res_r.bse[j]:>12.5f}")

    # --- discrete AME: hand-rolled fn vs statsmodels-params fed through same delta ---
    ame_hr_m, ci_hr_m = glm_ame_parity(x, g, y, cov_type="model")
    ame_hr_r, ci_hr_r = glm_ame_parity(x, g, y, cov_type="HC0")
    se_hr_m = (ci_hr_m[1] - ci_hr_m[0]) / (2 * 1.96)
    se_hr_r = (ci_hr_r[1] - ci_hr_r[0]) / (2 * 1.96)
    ame_sm_m, se_sm_m = _discrete_ame_and_se(np.asarray(res_m.params), exog,
                                             np.asarray(res_m.cov_params()))
    ame_sm_r, se_sm_r = _discrete_ame_and_se(np.asarray(res_r.params), exog,
                                             np.asarray(res_r.cov_params()))
    # statsmodels' own instantaneous (dydx) marginal effect, for reference
    marg = res_m.get_margeff(at="overall", method="dydx")
    dydx_g = marg.margeff[-1]

    print(f"\nDiscrete AME of G (estimand the experiment uses):")
    print(f"  hand-rolled : AME={ame_hr_m:+.6f}  model SE={se_hr_m:.6f}  HC0 SE={se_hr_r:.6f}")
    print(f"  statsmodels : AME={ame_sm_m:+.6f}  model SE={se_sm_m:.6f}  HC0 SE={se_sm_r:.6f}")
    print(f"  statsmodels get_margeff dydx (instantaneous, NOT discrete): {dydx_g:+.6f}")
    print(f"\n  -> model vs HC0 SE ratio (hand-rolled): {se_hr_r/se_hr_m:.3f} "
          f"(>1 means the sandwich is wider -> model SE was optimistic)")


def show_separation():
    print("\n" + "=" * 78)
    print("2. SEPARATION on the well-specified feature set (noiseless DGP)")
    print("   Setting 3, GLM design = [1, x^2, G]  (true boundary IS in x^2)")
    print("=" * 78)
    rng = np.random.default_rng(11)
    x, g, y, _ = setting3_draw(4000, rng)
    feats = np.hstack([x ** 2, g.reshape(-1, 1)])
    exog = sm.add_constant(feats)

    beta_hr, D, p_hr, *_ = glm_coef_cov(x ** 2, g, y)
    print(f"  hand-rolled (sklearn) : |beta|={np.linalg.norm(beta_hr):.1f}, "
          f"saturated p frac={np.mean((p_hr<1e-4)|(p_hr>1-1e-4)):.2f}, "
          f"train acc={(p_hr.round()==y).mean():.4f}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn separation/convergence warnings into errors
            res = sm.Logit(y, exog).fit(disp=0)
        print(f"  statsmodels MLE       : converged; max|beta|={np.max(np.abs(res.params)):.1f} "
              f"(huge -> diverging), max bse={np.max(res.bse):.1f}")
    except Exception as e:
        print(f"  statsmodels MLE       : raised {type(e).__name__}: {str(e).splitlines()[0]}")
    print("  -> the 'well-specified' GLM is a degenerate (separated) fit; its inference"
          "\n     is not on solid footing without a separation guard or outcome noise.")


def main():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*penalty.*")
        cross_check_nonseparable()
    show_separation()


if __name__ == "__main__":
    main()
