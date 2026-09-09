#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=eif_regeneration
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/jrudoler/tl_fairness
#SBATCH --output=slurm/logs/%x-%j.log
#SBATCH --error=slurm/logs/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/experiments/out/uv-cache"
export MPLCONFIGDIR="$PWD/experiments/out/mpl-cache"
workers="${SLURM_CPUS_PER_TASK:-1}"
outdir=experiments/out/eif_regeneration
case "$1" in
    trainsize) uv run --no-sync .venv/bin/python experiments/exp5_trainsize_coverage.py --n-jobs "$workers" --output "$outdir/trainsize_coverage.csv" --figure "$outdir/trainsize_coverage.png" ;;
    exp1) uv run --no-sync .venv/bin/python experiments/exp1_parity_coverage.py --n-jobs "$workers" --output "$outdir/exp1_parity_coverage.csv" --figure "$outdir/exp1_parity_coverage.png" ;;
    exp2) uv run --no-sync .venv/bin/python experiments/exp2_misspec_coverage.py --n-jobs "$workers" --output "$outdir/exp2_misspec_coverage.csv" --figure "$outdir/exp2_misspec_coverage.png" ;;
    robust) uv run --no-sync .venv/bin/python analysis/sim_robust/run.py --output "$outdir/robust_res.csv" ;;
    tmle) uv run --no-sync .venv/bin/python analysis/sim_tmle/run.py --output "$outdir/tmle_coverage.csv" ;;
    adult|law) uv run --no-sync .venv/bin/python analysis/eif_regeneration/paired_data.py --dataset "$1" --n-jobs "$(( workers < 3 ? workers : 3 ))" --output-dir "$outdir" ;;
    *) exit 2 ;;
esac
