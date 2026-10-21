#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=cmi_figure2_protocol
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/jrudoler/tl_fairness
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/logs/matplotlib" XDG_CACHE_HOME="$PWD/logs/cache"
.venv/bin/snakemake -s workflow/cmi_figure2_review.smk --cores "$SLURM_CPUS_PER_TASK" --quiet
