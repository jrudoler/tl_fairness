"""Figure 7: CMI coverage, one-step vs TMLE iteration vs boundary CI.

A row of coverage heatmaps (one per estimator) over (sample size, c), plus a
signed-error line plot vs c faceted by sample size. Directly answers whether the
TMLE iteration repairs the small-c coverage failure of Figure 4.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tlfair.plotting import configure_matplotlib, FULL_WIDTH

LABELS = {
    'one_step': 'One-step',
    'tmle_notarget': 'Substitution (no iter.)',
    'tmle': 'TMLE (iterated)',
    'tmle_bdry': 'TMLE + boundary CI',
}
ORDER = ['one_step', 'tmle_notarget', 'tmle', 'tmle_bdry']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/generated/cmi_tmle_coverage.csv')
    parser.add_argument('--coverage-output',
                        default='results/figures/fig7_cmi_tmle_coverage.pdf')
    parser.add_argument('--error-output',
                        default='results/figures/fig7_cmi_tmle_error.pdf')
    args = parser.parse_args()

    configure_matplotlib()
    df = pd.read_csv(args.input)
    estimators = [e for e in ORDER if e in df['estimator'].unique()]

    # --- Coverage heatmaps, one per estimator, shared 0-1 color scale ---
    fig, axs = plt.subplots(1, len(estimators),
                            figsize=(FULL_WIDTH * 1.9, 3.6), sharey=True)
    if len(estimators) == 1:
        axs = [axs]
    for i, e in enumerate(estimators):
        piv = (df[df['estimator'] == e]
               .pivot(index='sample_size', columns='c', values='coverage'))
        sns.heatmap(piv, ax=axs[i], vmin=0, vmax=1, cmap='viridis',
                    cbar=(i == len(estimators) - 1),
                    cbar_kws={'label': 'coverage'})
        axs[i].set_title(LABELS.get(e, e), fontsize=10)
        axs[i].set_xlabel('c')
        axs[i].set_ylabel('Sample Size' if i == 0 else '')
    fig.suptitle('95% CI coverage of CMI (nominal 0.95)', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.coverage_output)
    plt.close(fig)
    print(f'Wrote {args.coverage_output}', flush=True)

    # --- Signed error vs c, faceted by sample size, one line per estimator ---
    df2 = df.copy()
    df2['Estimator'] = df2['estimator'].map(lambda e: LABELS.get(e, e))
    g = sns.FacetGrid(df2, col='sample_size', col_wrap=3, hue='Estimator',
                      hue_order=[LABELS[e] for e in estimators])
    g.map_dataframe(sns.lineplot, x='c', y='error')
    g.map(plt.axhline, y=0, color='black', linestyle='--')
    g.add_legend()
    g.savefig(args.error_output)
    plt.close(g.figure)
    print(f'Wrote {args.error_output}', flush=True)


if __name__ == '__main__':
    main()
