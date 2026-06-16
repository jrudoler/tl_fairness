#!/bin/bash
#SBATCH --job-name=tlfair-analyze
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
#SBATCH --cpus-per-task=16          # perm-importance + CMI sims parallelise over these
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=whartonstat
#SBATCH --account=whartonstat

# Full-data real analysis (Adult + Law) and Table 1, to compare against the
# paper's Table 1. Run from the repo root on the cluster:
#     cd /home/jrudoler/tl_fairness
#     mkdir -p slurm/logs
#     sbatch slurm/analyze_fulldata.sbatch
#
# NOTE: metric_map now includes the TMLE variants, so analyze_adult/analyze_law
# fit 8 metrics each (5 one-step + 3 TMLE). Permutation feature-importance
# (Figure 5) is OFF by default, so this job is inference-only and produces
# Table 1. To also build the importance figure (hours of extra compute), add
# `fig5_importance` to the targets and `importance_samples=1000` to --config.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# --- environment -----------------------------------------------------------
# Use the .venv prebuilt on the (shared) login node. The compute node may have
# no outbound internet (TLS interception breaks pip/urllib here), so we do NOT
# run `uv sync` or any network step in the job; build the env on the login node
# first:  uv sync   (and curl the raw CSVs into data/raw/).
if [ ! -x .venv/bin/python ]; then
    echo "ERROR: .venv missing. On the login node run: uv sync" >&2
    exit 1
fi
NJOBS="${SLURM_CPUS_PER_TASK:-1}"
export TLFAIR_PYTHON=".venv/bin/python"

# --- run -------------------------------------------------------------------
# Raw CSVs must already exist in data/raw/ (download on the login node via curl,
# since the compute node has no working TLS to GitHub).
# NOTE: Snakemake 9's `--config` greedily consumes all following tokens as
# name=value pairs, so the rule targets MUST come before `--config`.
.venv/bin/python -m snakemake \
    analyze_adult analyze_law table1_inference \
    --cores "$NJOBS" --config njobs="$NJOBS"

echo "Done. Compare results/data/table1_inference.csv against the paper's Table 1."

# ---------------------------------------------------------------------------
# Faster inference-only comparison (skips the 1000-permutation importance and
# the TMLE metrics; fits each nuisance once per metric). Run directly, not via
# the protected Snakemake rule:
#
#   .venv/bin/python analysis/analyze_adult/run.py \
#       --input data/raw/adult.csv --output /tmp/adult_inf.pkl \
#       --metrics parity prob_parity opportunity prob_opp cmi \
#       --importance-samples 1 --n-jobs "$NJOBS"
#   .venv/bin/python analysis/analyze_law/run.py \
#       --input data/raw/law.csv --output /tmp/law_inf.pkl \
#       --metrics parity prob_parity opportunity prob_opp cmi \
#       --importance-samples 1 --n-jobs "$NJOBS"
#   .venv/bin/python analysis/table1_inference/run.py \
#       --adult-input /tmp/adult_inf.pkl --law-input /tmp/law_inf.pkl \
#       --csv-output /tmp/table1.csv --tex-output /tmp/table1.tex
# ---------------------------------------------------------------------------
