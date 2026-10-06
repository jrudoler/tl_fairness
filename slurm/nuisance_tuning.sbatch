#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=fairness_cv
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-279%24
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/cache/mpl" XDG_CACHE_HOME="$PWD/cache"
PYTHON="${TLFAIR_PYTHON:-/home/jrudoler/inductive-bias/.venv/bin/python}"
sizes=(10000 5000 2500 1000 500 250 100)
configuration=$((SLURM_ARRAY_TASK_ID / 40))
shard=$((SLURM_ARRAY_TASK_ID % 40))
size=${sizes[$configuration]}
"$PYTHON" experiments/nuisance_tuning.py run --train-size "$size" \
    --rep-start "$((shard * 25))" --reps 25 --n-jobs "$SLURM_CPUS_PER_TASK" \
    --output-dir "data/generated/nuisance_tuning_shards/primary_${size}_${shard}"
