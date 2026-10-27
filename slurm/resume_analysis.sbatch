#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=fairness-resume-nuisance
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/jrudoler/tl_fairness
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/logs/cache/matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$PWD/logs/cache/slurm-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
.venv/bin/python analysis/tools/resume_analysis.py \
    --run-id "${1:?Supply the interrupted run ID}" \
    --analysis "${2:?Supply the nuisance analysis ID}"
