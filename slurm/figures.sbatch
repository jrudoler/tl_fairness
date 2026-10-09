#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=tlfair-render
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
# Render from verified publication inputs; this target never starts simulations.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tlfair-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/tlfair-cache-${SLURM_JOB_ID}"
"${TLFAIR_PYTHON:-.venv/bin/python}" -m snakemake all --cores 1
