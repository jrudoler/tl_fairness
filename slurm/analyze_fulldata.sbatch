#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=tlfair-real-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
# Usage: mkdir -p logs; sbatch slurm/analyze_fulldata.sbatch NEW_RUN_ID
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tlfair-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tlfair-cache-${SLURM_JOB_ID}"
python_bin="${TLFAIR_PYTHON:-.venv/bin/python}"
for dataset in adult law; do
    "$python_bin" analysis/tools/run_analysis.py --analysis "real_data_$dataset" \
        --run-id "${1:?Supply a fresh run ID}" --n-jobs 1
done
