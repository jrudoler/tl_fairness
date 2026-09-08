"""Figure 4: CMI error (by kappa, faceted on sample size) and TL coverage heatmap.

Produces two PDFs: the per-estimator error line plots and the TL coverage
heatmap (paper Section 4.2, Figure 5). Both are sized in physical inches to
match how they are placed side by side in the manuscript (0.45\\linewidth and
0.5\\linewidth of FULL_WIDTH, respectively) so neither is authored far larger
than its final rendered size -- oversizing a source figure and relying on
LaTeX to shrink it is what made the line-plot grid tiny and unreadable.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tlfair.plotting import configure_matplotlib, FULL_WIDTH

KAPPA_LABEL = r'$\kappa$'


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
    # Compare only the targeted-learning (TL) and KNN estimators; the legacy
    # separate-marginals variant (TL-sep) was dropped.
    cmp = cmp[cmp['type'].isin(['TL', 'KNN'])]
    with open(args.truth_input, 'rb') as f:
        truth = pickle.load(f)
    truth_f = {float(k): v for k, v in truth.items()}
    cmp['error'] = cmp['mean'] - cmp['kappa'].astype(float).map(truth_f)

    # Sized (height, aspect) so the 3-across grid renders at roughly the same
    # physical scale as the coverage heatmap below, instead of the seaborn
    # default (~10in square) that LaTeX then had to shrink to ~3in -- crushing
    # the text -- to fit next to the heatmap (see module docstring).
    g = sns.FacetGrid(
        cmp, col='sample size', col_wrap=3, hue='type',
        height=1.5, aspect=1.05, sharex=True, sharey=True,
    )
    g.map_dataframe(sns.lineplot, x='kappa', y='error', marker='o', markersize=3)
    g.set_titles(template='n = {col_name}', size=9)
    g.set_axis_labels(KAPPA_LABEL, 'Error')
    # refline draws the y=0 reference on every facet without joining the hue
    # legend (g.map(axhline) would, turning every legend entry into a black dash).
    g.refline(y=0, color='black', linestyle='--', linewidth=0.8)
    for ax in g.axes.flat:
        ax.tick_params(labelsize=8)
    g.add_legend(title='Estimator', fontsize=9, title_fontsize=9)
    g.savefig(args.error_output)
    plt.close(g.figure)
    print(f'Wrote {args.error_output}', flush=True)

    # --- TL coverage heatmap over (sample size, kappa) ---
    cov = pd.read_csv(args.coverage_input)
    plot = cov.pivot(index='sample_size', columns='kappa', values='coverage')
    plot.columns = [f'{c:g}' for c in plot.columns]
    fig, ax = plt.subplots(
        figsize=(0.5 * FULL_WIDTH * 1.6, 0.5 * FULL_WIDTH * 1.6 * 0.77)
    )
    sns.heatmap(plot, ax=ax, cbar_kws={'label': 'Coverage'})
    ax.set_ylabel('Sample Size')
    ax.set_xlabel(KAPPA_LABEL)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(args.coverage_output)
    print(f'Wrote {args.coverage_output}', flush=True)


if __name__ == '__main__':
    main()
