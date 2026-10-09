#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=nuisance_collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/collect-%j.log
#SBATCH --error=logs/collect-%j.log
# Historical shard-run launcher: use only with its matching frozen source snapshot.
# For new configured runs, use slurm/analysis.sbatch; see EXPERIMENTS.md.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/cache/mpl" XDG_CACHE_HOME="$PWD/cache"
PYTHON="${TLFAIR_PYTHON:-/home/jrudoler/inductive-bias/.venv/bin/python}"
"$PYTHON" experiments/nuisance_training.py collect \
    --input-dir data/generated/nuisance_training_shards \
    --output-dir data/generated/nuisance_training
"$PYTHON" analysis/nuisance_tables/validate.py data/generated/nuisance_training
