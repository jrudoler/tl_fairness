#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=fairness_rate
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-639%24
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/cache/mpl" XDG_CACHE_HOME="$PWD/cache"
PYTHON="${TLFAIR_PYTHON:-/home/jrudoler/inductive-bias/.venv/bin/python}"
"$PYTHON" experiments/nuisance_matched_rate.py run --array-index "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
