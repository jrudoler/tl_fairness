#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=fairness_early
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-319%24
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tlfair-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tlfair-cache-${SLURM_JOB_ID}"
# Run from the isolated snapshot directory; no edits to the remote checkout.
PYTHON="${TLFAIR_PYTHON:-/home/jrudoler/inductive-bias/.venv/bin/python}"
"$PYTHON" analysis/sim_nuisance_early_stopping/run.py --array-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "$SLURM_CPUS_PER_TASK" --output-dir data/generated/nuisance_early_stopping_shards
