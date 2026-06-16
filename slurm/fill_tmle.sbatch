#!/bin/bash
#SBATCH --job-name=tlfair-tmle
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat

# Fill the 3 TMLE inference rows into the already-persisted (inference-only)
# analyze pickles and regenerate Table 1. No permutation importance.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if [ ! -x .venv/bin/python ]; then
    echo "ERROR: .venv missing. On the login node run: uv sync" >&2
    exit 1
fi

PYTHONPATH=. .venv/bin/python slurm/fill_tmle.py
echo "Done. Table 1 now includes the TMLE rows."
