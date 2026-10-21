#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=cmi_learner_review
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-29%4
#SBATCH --chdir=/home/jrudoler/tl_fairness
#SBATCH --output=logs/slurm/%x-%A_%a.log
#SBATCH --error=logs/slurm/%x-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/logs/matplotlib" XDG_CACHE_HOME="$PWD/logs/cache"
kappas=(0 2.5 4)
cell=$((SLURM_ARRAY_TASK_ID / 10))
shard=$((SLURM_ARRAY_TASK_ID % 10))
.venv/bin/python analysis/cmi_learner_review/run.py run \
    --kappa "${kappas[$cell]}" --rep-start "$((10 * shard))" --reps 10 \
    --n-jobs "$SLURM_CPUS_PER_TASK"
