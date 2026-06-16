"""Figure 4: CMI error (by c, faceted on sample size) and TL coverage heatmap.

Produces two PDFs: the per-estimator error line plots and the TL coverage
heatmap (paper Section 4.2).
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tlfair.plotting import configure_matplotlib, FULL_WIDTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage-input', default='data/generated/cmi_coverage.csv')
    parser.add_argument('--compare-input', default='data/generated/cmi_compare.csv')
    parser.add_argument('--truth-input', default='data/generated/truth_dict.pkl')
    parser.add_argument('--error-output', default='results/figures/fig4_cmi_error.pdf')
    parser.add_argument('--coverage-output', default='results/figures/fig4_cmi_coverage.pdf')
    args = parser.parse_args()

    configure_matplotlib()

    # --- Error line plots, faceted by sample size, one line per estimator ---
    cmp = pd.read_csv(args.compare_input)
    with open(args.truth_input, 'rb') as f:
        truth = pickle.load(f)
    truth_f = {float(k): v for k, v in truth.items()}
    cmp['error'] = cmp['mean'] - cmp['c'].astype(float).map(truth_f)

    g = sns.FacetGrid(cmp, col='sample size', col_wrap=3, hue='type')
    g.map_dataframe(sns.lineplot, x='c', y='error')
    g.map(plt.axhline, y=0, color='black', linestyle='--')
    g.add_legend()
    g.savefig(args.error_output)
    plt.close(g.figure)
    print(f'Wrote {args.error_output}', flush=True)

    # --- TL coverage heatmap over (sample size, c) ---
    cov = pd.read_csv(args.coverage_input)
    plot = cov.pivot(index='sample_size', columns='c', values='coverage')
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4))
    sns.heatmap(plot, ax=ax)
    ax.set_ylabel('Sample Size')
    ax.set_xlabel('c')
    fig.tight_layout()
    fig.savefig(args.coverage_output)
    print(f'Wrote {args.coverage_output}', flush=True)


if __name__ == '__main__':
    main()
