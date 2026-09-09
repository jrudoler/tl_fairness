"""Compare old and corrected intervals on identical robustness fits and draws."""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.sim_robust.run import robust_exp
from tlfair import metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/out/eif_regeneration'))
    args = parser.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # The original entrypoint left GB's tie-breaking seed unspecified.
    np.random.seed(123)
    original = metrics._prob_group_contrast
    records = []

    def compare(predictions: np.ndarray, residual: np.ndarray, weights: np.ndarray,
                strata: np.ndarray) -> tuple[float, tuple[float, float]]:
        result = original(predictions, residual, weights, strata)
        contributions = (weights * residual[:, None] + strata * predictions[:, None]) / strata.mean(axis=0)
        old = metrics._wald_ci(result[0], contributions[:, 1] - contributions[:, 0] - result[0])
        records.append({'estimate': result[0], 'old_lo': old[0], 'old_hi': old[1],
                        'new_lo': result[1][0], 'new_hi': result[1][1]})
        return result

    metrics._prob_group_contrast = compare
    sizes, reps = [250, 500, 1000, 2500], 100
    result = robust_exp(sizes, reps, seed=123, truth_n=10_000_000)
    result.to_csv(out/'robust_res.csv', index=False)
    raw = pd.DataFrame(records)
    raw['sample_size'] = np.repeat(sizes, reps * 4)
    raw['cases'] = np.tile(['misspecified', 'outcome_correct', 'propensity_correct', 'both_correct'], reps * len(sizes))
    raw['replicate'] = np.tile(np.repeat(np.arange(reps), 4), len(sizes))
    truth = (result.mean_estimate - result.error).iloc[0]
    for version in ['old', 'new']:
        raw[f'{version}_coverage'] = (raw[f'{version}_lo'] <= truth) & (truth <= raw[f'{version}_hi'])
        raw[f'{version}_width'] = raw[f'{version}_hi'] - raw[f'{version}_lo']
    raw.to_csv(out/'robust_paired_replicates.csv', index=False)
    summary = raw.groupby(['sample_size', 'cases'])[['old_coverage', 'new_coverage', 'old_width', 'new_width']].mean()
    summary.to_csv(out/'robust_paired.csv')
    logging.info('Completed paired EIF comparison\n%s', summary)


if __name__ == '__main__':
    main()
