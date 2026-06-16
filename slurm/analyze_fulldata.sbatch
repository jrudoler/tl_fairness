#!/bin/bash
#SBATCH --job-name=tlfair-analyze
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out
#SBATCH --cpus-per-task=16          # perm-importance + CMI sims parallelise over these
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=whartonstat
#SBATCH --account=jrudoler

# Full-data real analysis (Adult + Law) and Table 1, to compare against the
# paper's Table 1. Run from the repo root on the cluster:
#     cd /home/jrudoler/tl_fairness
#     mkdir -p slurm/logs
#     sbatch slurm/analyze_fulldata.sbatch
#
# NOTE: metric_map now includes the TMLE variants, so analyze_adult/analyze_law
# fit 8 metrics each (5 one-step + 3 TMLE). For a quick *inference-only* check
# against Table 1, see the scoped commands at the bottom of this file.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

# --- environment -----------------------------------------------------------
# uv is the project's runner. If it isn't on PATH, install once (login node,
# needs internet) or load your site's module:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   # or: module load uv   (site-dependent)
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen          # builds .venv from uv.lock (base deps; no JAX needed)

NJOBS="${SLURM_CPUS_PER_TASK:-1}"
export TLFAIR_PYTHON=".venv/bin/python"

# --- run -------------------------------------------------------------------
# download_* rules fetch the raw CSVs if data/raw is empty (login node has net;
# if compute nodes don't, run the two download rules on the login node first).
uv run snakemake --cores "$NJOBS" --config njobs="$NJOBS" \
    analyze_adult analyze_law table1_inference fig5_importance

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
