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
    # Back-compat: a single-benchmark CSV with no truth_type column is treated
    # as the conditional (honest) benchmark.
    if 'truth_type' not in df.columns:
        df['truth_type'] = 'cond'
    truth_rows = [('uncond', 'Unconditional MI benchmark (mismatched)'),
                  ('cond', 'Conditional CMI benchmark (honest)')]
    truth_rows = [(t, lab) for t, lab in truth_rows
                  if t in df['truth_type'].unique()]

    # --- Coverage heatmaps: rows = benchmark, cols = estimator, shared scale ---
    nrow, ncol = len(truth_rows), len(estimators)
    fig, axs = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH * 1.9, 3.4 * nrow),
                            squeeze=False)
    for r, (tt, row_label) in enumerate(truth_rows):
        for i, e in enumerate(estimators):
            piv = (df[(df['estimator'] == e) & (df['truth_type'] == tt)]
                   .pivot(index='sample_size', columns='c', values='coverage'))
            sns.heatmap(piv, ax=axs[r][i], vmin=0, vmax=1, cmap='viridis',
                        cbar=(i == ncol - 1), cbar_kws={'label': 'coverage'})
            if r == 0:
                axs[r][i].set_title(LABELS.get(e, e), fontsize=10)
            axs[r][i].set_xlabel('c')
            axs[r][i].set_ylabel(f'{row_label}\nSample Size' if i == 0 else '')
    fig.suptitle('95% CI coverage of CMI (nominal 0.95)', fontsize=11)
    fig.tight_layout()
    fig.savefig(args.coverage_output)
    plt.close(fig)
    print(f'Wrote {args.coverage_output}', flush=True)

    # --- Signed error vs c (honest conditional benchmark), faceted by size ---
    df2 = df[df['truth_type'] == 'cond'].copy()
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
