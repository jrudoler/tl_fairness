"""Reproduce the matched-rate replacement using the cluster checkpoint boundaries."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.nuisance_matched_rate import grid, run_shard, collect

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-jobs',type=int,default=1)
    p.add_argument('--output-dir',type=Path,default=Path('data/generated/nuisance_matched_rate_shards'))
    p.add_argument('--summary-dir',type=Path,default=Path('data/generated/nuisance_matched_rate'))
    a=p.parse_args()
    for s,n in grid():
        for shard in range(40):
            run_shard(a.output_dir/f'{s}_{n}_{shard:02d}',s,n,rep_start=shard*25,n_jobs=a.n_jobs)
    collect(a.output_dir,'data/generated/nuisance_training',a.summary_dir)
