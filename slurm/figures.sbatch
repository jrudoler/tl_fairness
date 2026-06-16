#!/bin/bash
#SBATCH --job-name=tlfair-figs
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat

# Regenerate all paper figures and tables via `snakemake all`. Feature-importance
# (fig5) is OFF by default, so this runs the 4 simulations (sim_parity, sim_robust,
# sim_tmle, sim_cmi) and builds fig1-4, fig6, table1, table2. The analyze pickles
# already exist, so analyze_adult/analyze_law are skipped.
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
