#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=parity-retraining
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
# Usage: mkdir -p logs; sbatch slurm/parity_retraining.sbatch NEW_RUN_ID
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tlfair-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tlfair-cache-${SLURM_JOB_ID}"
"${TLFAIR_PYTHON:-.venv/bin/python}" analysis/tools/run_analysis.py \
    --analysis parity_retraining --run-id "${1:?Supply a fresh run ID}" \
    --n-jobs "$SLURM_CPUS_PER_TASK"
