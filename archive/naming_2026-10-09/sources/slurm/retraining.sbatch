#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=fairness_retraining
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/jrudoler/tl_fairness
#SBATCH --output=slurm/logs/%x-%j.log
#SBATCH --error=slurm/logs/%x-%j.log

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$PWD/experiments/out/cache"
export MPLCONFIGDIR="$PWD/experiments/out/mpl-cache"
export UV_CACHE_DIR="$PWD/experiments/out/uv-cache"

if [[ "${1:-}" == "independence" ]]; then
    uv run --no-sync .venv/bin/python experiments/exp6_retraining.py \
        --effects 0 --association 0 --n-jobs "$SLURM_CPUS_PER_TASK" \
        --output-dir experiments/out/retraining_independent_dense
else
    uv run --no-sync .venv/bin/python -m snakemake sim_retraining \
        --cores "$SLURM_CPUS_PER_TASK" --config njobs="$SLURM_CPUS_PER_TASK" --rerun-incomplete
fi
