#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=tlfair-analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
# Usage: mkdir -p logs; sbatch slurm/analysis.sbatch ANALYSIS NEW_RUN_ID [REFERENCE_DIR]
# The configured run may outlast this allocation; choose resources for the analysis.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tlfair-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tlfair-cache-${SLURM_JOB_ID}"
python_bin="${TLFAIR_PYTHON:-.venv/bin/python}"
args=(--analysis "${1:?Supply an analysis ID}" --run-id "${2:?Supply a fresh run ID}"
      --n-jobs "$SLURM_CPUS_PER_TASK" --compare-mem-gb 24)
if [[ -n "${3:-}" ]]; then args+=(--reference-dir "$3"); fi
"$python_bin" analysis/tools/run_analysis.py "${args[@]}"
