"""Run the prespecified matched nuisance-learning grid, optionally as array shards."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.nuisance_early_stopping import SCENARIOS, SIZES, CONTROL_SIZES, run_shard, collect


def grid():
    # Largest fits first, so expensive jobs do not leave a serial tail.
    return [(s, n) for n in reversed(SIZES) for s in SCENARIOS
            if s == 'primary' or n in CONTROL_SIZES]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--array-index', type=int)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--reps', type=int, default=1000)
    p.add_argument('--output-dir', type=Path, default=Path('data/generated/nuisance_early_stopping_shards'))
    p.add_argument('--summary-dir', type=Path, default=Path('data/generated/nuisance_early_stopping'))
    args = p.parse_args()
    if args.array_index is not None:
        if args.reps != 1000 or not 0 <= args.array_index < 320:
            p.error('Array uses exactly 320 shards of 50 replicates')
        configuration, shard = divmod(args.array_index, 20)
        scenario, size = grid()[configuration]
        run_shard(args.output_dir/f'{scenario}_{size}_{shard:02d}', scenario, size,
                  reps=50, rep_start=50*shard, n_jobs=args.n_jobs)
    else:
        for scenario, size in grid():
            run_shard(args.output_dir/f'{scenario}_{size}_full', scenario, size,
                      reps=args.reps, n_jobs=args.n_jobs)
        collect(args.output_dir, args.summary_dir, args.reps)


if __name__ == '__main__':
    main()
