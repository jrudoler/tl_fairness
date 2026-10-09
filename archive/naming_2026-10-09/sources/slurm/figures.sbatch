#!/bin/bash
#SBATCH --job-name=tlfair-figs
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat

# Build current manuscript results via `snakemake all`. The active analyses are
# retraining, early-stopped nuisance learning (plus its fixed-tree reference),
# conditioning-set CMI, appendix CMI, and Adult/Law inference. Existing inputs
# are reused when current; retired parity/robustness/TMLE figures are excluded.
# Review a dry run first; missing inputs can launch expensive simulations.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if [ ! -x .venv/bin/python ]; then
    echo "ERROR: .venv missing. On the login node run: uv sync" >&2
    exit 1
fi
NJOBS="${SLURM_CPUS_PER_TASK:-1}"
export TLFAIR_PYTHON=".venv/bin/python"

# njobs parallelises the expensive sims; compare_mem_gb bounds sim_cmi's knncmi
# distance-array workers (~4 GB each) against the 64 GB allocation.
# Targets BEFORE --config (Snakemake 9 consumes all trailing tokens as config).
.venv/bin/python -m snakemake all \
    --cores "$NJOBS" --config njobs="$NJOBS" compare_mem_gb=32

echo "Done. New figures in results/figures/, tables in results/data/."
