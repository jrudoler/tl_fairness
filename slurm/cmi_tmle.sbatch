#!/bin/bash
#SBATCH --job-name=tlfair-cmitmle
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat

# CMI one-step vs TMLE-iteration coverage experiment: runs sim_cmi_tmle and
# builds fig7 (coverage heatmaps + signed-error line plot). No knncmi, so the
# memory footprint is modest.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if [ ! -x .venv/bin/python ]; then
    echo "ERROR: .venv missing. On the login node run: uv sync" >&2
    exit 1
fi
NJOBS="${SLURM_CPUS_PER_TASK:-1}"
export TLFAIR_PYTHON=".venv/bin/python"

.venv/bin/python -m snakemake fig7_cmi_tmle \
    --cores "$NJOBS" --config njobs="$NJOBS"

echo "Done. See results/figures/fig7_cmi_tmle_{coverage,error}.pdf"
