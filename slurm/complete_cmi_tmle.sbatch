#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=paper-cmi-tmle
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/jrudoler/tl_fairness
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python analysis/tools/complete_cmi_tmle.py "$@"
