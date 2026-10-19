"""Paired one-step/substitution/TMLE CMI simulation with a KNN baseline."""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tlfair.cmi_sim import _draw_until_valid, paired_cmi_sim, knncmi_sim, cmi_ground_truth
from tlfair.cmi_estimation import replicate_records, require_convergence

LOG = logging.getLogger(__name__)


def _safe_compare_jobs(n: int, n_jobs: int, mem_gb: float) -> int:
    per_worker_gb = 5*n**2*8/1e9
    return max(1, min(n_jobs, int(mem_gb/per_worker_gb))) if per_worker_gb > 0 else n_jobs


def _replicate(n: int, kappa: float, rng: np.random.Generator, replicate: int,
               compare: bool) -> list[dict]:
    result = _draw_until_valid(paired_cmi_sim, n, kappa, rng)
    rows = replicate_records(result, sample_size=n, kappa=kappa, replicate=replicate)
    if compare:
        rows.append({'sample_size':n, 'kappa':kappa, 'replicate':replicate,
                     'estimator':'knn', 'estimate':knncmi_sim(n, kappa, rng=rng)})
    return rows


def summarize_coverage(records: pd.DataFrame, truth: dict[float, float]) -> pd.DataFrame:
    records = records.copy()
    records['covered'] = ((records.ci_low <= records.kappa.map(truth)) &
                          (records.ci_high >= records.kappa.map(truth))).astype(float)
    result = records.groupby(['sample_size','kappa','estimator']).agg(
        coverage=('covered','mean'), estimate=('estimate','mean')).reset_index()
    result['error'] = result.estimate-result.kappa.map(truth)
    return result


def summarize_comparison(records: pd.DataFrame) -> pd.DataFrame:
    result = records.groupby(['sample_size','kappa','estimator']).estimate.agg(
        mean='mean', bottom_five=lambda x:x.quantile(.05), top_five=lambda x:x.quantile(.95)).reset_index()
    result['type'] = result.estimator.map({'tmle':'TMLE','one_step':'One-step','substitution':'Substitution','knn':'KNN'})
    return result.rename(columns={'sample_size':'sample size'})


def run_experiment(weights: list[float], sizes: list[int], n_truth: int = 1000000,
                   sims: int = 100, compare_repeats: int = 10, seed: int = 123,
                   conditional_truth: bool = True, n_jobs: int = 1,
                   compare_mem_gb: float = 16, records_dir: Path | None = None) -> tuple:
    def task_rng(kind: int, kappa: float, size: int = 0) -> np.random.Generator:
        bits = np.asarray([float(kappa)], dtype='<f8').view('<u4').ravel().tolist()
        return np.random.default_rng(np.random.SeedSequence([seed,kind,size,*bits]))

    coverage_records, comparison_records, truth, timings = [], [], {}, {}
    if records_dir:
        records_dir.mkdir(parents=True, exist_ok=True)
    for kappa in weights:
        start = time.perf_counter()
        truth[kappa] = cmi_ground_truth(kappa, 3, n_truth, task_rng(0,kappa), conditional=conditional_truth)
        timings[f'truth_{kappa}'] = time.perf_counter()-start
        for size in sizes:
            start = time.perf_counter()
            streams = task_rng(1,kappa,size).spawn(sims)
            batches = Parallel(n_jobs=n_jobs)(delayed(_replicate)(size,kappa,child,i,False)
                                              for i,child in enumerate(streams))
            coverage_records.extend(row for batch in batches for row in batch)
            timings[f'coverage_kappa={kappa}_n={size}'] = time.perf_counter()-start
            if records_dir:
                pd.DataFrame(coverage_records).to_csv(records_dir/'coverage_replicates.csv.gz', index=False)
            LOG.info('Coverage completed: kappa=%s n=%s, %.1fs', kappa,size,timings[f'coverage_kappa={kappa}_n={size}'])
    for size in sizes:
        for kappa in weights:
            start = time.perf_counter()
            streams = task_rng(2,kappa,size).spawn(compare_repeats)
            jobs = _safe_compare_jobs(size,n_jobs,compare_mem_gb)
            batches = Parallel(n_jobs=jobs)(delayed(_replicate)(size,kappa,child,i,True)
                                            for i,child in enumerate(streams))
            comparison_records.extend(row for batch in batches for row in batch)
            timings[f'comparison_kappa={kappa}_n={size}'] = time.perf_counter()-start
            if records_dir:
                pd.DataFrame(comparison_records).to_csv(records_dir/'comparison_replicates.csv.gz', index=False)
            LOG.info('Comparison completed: kappa=%s n=%s', kappa,size)
    coverage, comparison = pd.DataFrame(coverage_records), pd.DataFrame(comparison_records)
    require_convergence(coverage)
    require_convergence(comparison)
    return summarize_coverage(coverage,truth), summarize_comparison(comparison), truth, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', nargs='+', type=float, default=[0,.5,1,1.5,2,2.5,3,3.5,4])
    parser.add_argument('--sizes', nargs='+', type=int, default=[500,750,1000,1750,2500,3750,5000,7500,10000])
    parser.add_argument('--n-truth', type=int, default=1000000)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--compare-repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--unconditional-truth', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--compare-mem-gb', type=float, default=16)
    parser.add_argument('--coverage-output', type=Path, default=ROOT/'data/generated/cmi_estimation_coverage.csv')
    parser.add_argument('--compare-output', type=Path, default=ROOT/'data/generated/cmi_estimation_comparison.csv')
    parser.add_argument('--truth-output', type=Path, default=ROOT/'data/generated/cmi_population_truth.pkl')
    parser.add_argument('--timing-output', type=Path, default=ROOT/'data/generated/cmi_estimation_timing.pkl')
    parser.add_argument('--records-dir', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    coverage, compare, truth, timings = run_experiment(args.weights,args.sizes,args.n_truth,args.sims,
        args.compare_repeats,args.seed,not args.unconditional_truth,args.n_jobs,args.compare_mem_gb,args.records_dir)
    for path in [args.coverage_output,args.compare_output,args.truth_output,args.timing_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.coverage_output,index=False)
    compare.to_csv(args.compare_output,index=False)
    for path, value in [(args.truth_output,truth),(args.timing_output,timings)]:
        with path.open('wb') as stream:
            pickle.dump(value,stream)


if __name__ == '__main__':
    main()
