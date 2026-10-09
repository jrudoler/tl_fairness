#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=fairness_nuisance
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-319%16
#SBATCH --output=logs/%A_%a.log
#SBATCH --error=logs/%A_%a.log
# Historical shard-run launcher: use only with its matching frozen source snapshot.
# For new configured runs, use slurm/analysis.sbatch; see EXPERIMENTS.md.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/cache/mpl" XDG_CACHE_HOME="$PWD/cache"
# Run from the isolated snapshot directory; no edits to the remote checkout.
PYTHON="${TLFAIR_PYTHON:-/home/jrudoler/inductive-bias/.venv/bin/python}"
"$PYTHON" analysis/sim_nuisance_reference/run.py --array-index "$SLURM_ARRAY_TASK_ID" \
    --n-jobs "$SLURM_CPUS_PER_TASK" --output-dir data/generated/nuisance_training_shards
