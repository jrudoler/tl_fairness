"""Run the CV extension using the original draws, then collect matched results."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.nuisance_training import SIZES
from experiments.nuisance_tuning import run_shard,collect

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-jobs',type=int,default=1)
    p.add_argument('--shards',default='data/generated/nuisance_tuning_shards')
    p.add_argument('--base-dir',default='data/generated/nuisance_training')
    p.add_argument('--output-dir',default='data/generated/nuisance_training_cv')
    a=p.parse_args()
    # Match cluster shard boundaries so completed records can resume locally.
    for n in reversed(SIZES):
        for shard in range(40):
            run_shard(Path(a.shards)/f'primary_{n}_{shard}',n,shard*25,25,a.n_jobs)
    collect(a.shards,a.base_dir,a.output_dir)
