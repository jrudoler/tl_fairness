"""Distinguish DR consistency from Wald coverage in the Setting 3 DGP.

Diagnostic controls, separate from the published boosting experiment. The
quadratic logistic fits contain the exact generating law; the true-probability
controls have no estimation error in the named nuisance.
"""
import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special
from scipy.stats import qmc
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tlfair.metrics import _prob_group_contrast
from tlfair.simulations import (
    SETTING3_GROUP_COEF, SETTING3_OUTCOME_COEF, setting3_draw,
)


def population_truth() -> tuple[float, float]:
    """Independent scrambled Sobol integrations, averaging out G and Y noise."""
    values = []
    for seed in range(8):
        x = special.ndtri(qmc.Sobol(5, scramble=True, seed=3100 + seed).random_base2(18))
        d, pi = special.expit(x**2 @ SETTING3_OUTCOME_COEF), special.expit(x**2 @ SETTING3_GROUP_COEF)
        values.append(np.mean(pi*d)/pi.mean() - np.mean((1-pi)*d)/(1-pi).mean())
    return float(np.mean(values)), float(np.std(values, ddof=1)/np.sqrt(len(values)))


def replicate(n_train: int, n_eval: int, rep: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, n_train, n_eval, rep]))
    x, g, y, d = setting3_draw(n_train+n_eval, rng, bernoulli=True, bernoulli_group=True)
    xt, xe = x[:n_train], x[n_train:]
    d, pi = d[n_train:], special.expit(xe**2 @ SETTING3_GROUP_COEF)
    fitted = {}
    for name, response in [('d', y), ('pi', g)]:
        for form in ['linear', 'quadratic']:
            tr, ev = (xt, xe) if form == 'linear' else (xt**2, xe**2)
            # Negligible penalty approximates the correctly specified MLE.
            model = LogisticRegression(C=1e8, solver='lbfgs', max_iter=1000, tol=1e-9)
            model.fit(tr, response[:n_train])
            if model.n_iter_[0] >= 1000:
                raise RuntimeError('Logistic regression failed to converge')
            fitted[name, form] = model.predict_proba(ev)[:, 1]
    cases = {
        'both_true': (d, pi),
        'true_outcome_linear_group': (d, fitted['pi', 'linear']),
        'linear_outcome_true_group': (fitted['d', 'linear'], pi),
        'quadratic_outcome_linear_group': (fitted['d', 'quadratic'], fitted['pi', 'linear']),
        'linear_outcome_quadratic_group': (fitted['d', 'linear'], fitted['pi', 'quadratic']),
        'both_quadratic': (fitted['d', 'quadratic'], fitted['pi', 'quadratic']),
    }
    strata = np.column_stack([g[n_train:] == 0, g[n_train:] == 1])
    rows = []
    for case, (dh, ph) in cases.items():
        estimate, (lo, hi) = _prob_group_contrast(dh, y[n_train:]-dh, np.column_stack([1-ph, ph]), strata)
        # Independent evaluation X also estimates the conditional product drift.
        drift = -np.mean((ph-pi)*(dh-d))*(1/pi.mean()+1/(1-pi).mean())
        rows.append(dict(n_train=n_train, n_eval=n_eval, replicate=rep, case=case,
                         estimate=estimate, lo=lo, hi=hi, se=(hi-lo)/3.92,
                         product_drift=drift))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type=int, nargs='+', default=[1250, 12500])
    parser.add_argument('--n-eval', type=int, default=1250)
    parser.add_argument('--reps', type=int, default=500)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/out/eif_regeneration/dr_controls'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    truth, integration_se = population_truth()
    logging.info('Population disparity %.9f; integration SE %.9f', truth, integration_se)
    pd.DataFrame([dict(truth=truth, integration_se=integration_se, seed=args.seed,
                       reps=args.reps)]).to_csv(out/'metadata.csv', index=False)
    all_rows = []
    for n_train in args.n_train:
        batches = Parallel(n_jobs=args.n_jobs)(delayed(replicate)(n_train, args.n_eval, r, args.seed) for r in range(args.reps))
        all_rows.extend(row for batch in batches for row in batch)
        logging.info('Completed %s training / %s evaluation: %s replicates', n_train, args.n_eval, args.reps)
    raw = pd.DataFrame(all_rows)
    raw['error'] = raw.estimate-truth
    raw['covered'] = (raw.lo <= truth) & (truth <= raw.hi)
    raw.to_csv(out/'replicates.csv', index=False)
    summary = raw.groupby(['n_train', 'n_eval', 'case']).agg(
        bias=('error', 'mean'), sd=('estimate', 'std'), mean_se=('se', 'mean'),
        coverage=('covered', 'mean'), mean_product_drift=('product_drift', 'mean'))
    summary.to_csv(out/'summary.csv')
    logging.info('Results\n%s', summary.to_string())


if __name__ == '__main__':
    main()
