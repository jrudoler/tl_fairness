"""Figure 1: demographic parity estimates vs sample size (paper Section 4.1).

Top panel: one estimate per simulated dataset across a dense log-spaced grid
(the funnel). Bottom panel: estimate with a 95% Wald CI band on a coarser
subsample of sizes. Dotted line is the true parity value.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import matplotlib.pyplot as plt

from tlfair.plotting import configure_matplotlib, FULL_WIDTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/generated/parity_fig1.csv')
    parser.add_argument('--output', default='results/figures/fig1_parity.pdf')
    parser.add_argument('--ci-stride', type=int, default=10,
                        help='subsample factor for the CI panel')
    args = parser.parse_args()

    configure_matplotlib()
    df = pd.read_csv(args.input).sort_values('sample_size').reset_index(drop=True)
    truth = df['truth'].iloc[0]
    ci = df.iloc[::args.ci_stride]

    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 5), sharex=True)

    axes[0].scatter(df['sample_size'], df['estimate'], s=12, alpha=0.7)
    axes[0].axhline(truth, color='black', linestyle='--')
    axes[0].set_ylabel('Estimate')

    axes[1].errorbar(ci['sample_size'], ci['estimate'],
                     yerr=1.96 * ci['std'], fmt='o', markersize=4, capsize=2)
    axes[1].axhline(truth, color='black', linestyle='--')
    axes[1].set_xlabel('Sample size')
    axes[1].set_ylabel('Estimate')

    # Log x-axis fit tightly to the data range (no padding to 10^0).
    for ax in axes:
        ax.set_xscale('log')
        ax.margins(x=0)
        ax.autoscale(enable=True, axis='x', tight=True)

    fig.tight_layout()
    fig.savefig(args.output)
    print(f'Wrote {args.output} ({len(df)} points, {len(ci)} with CIs)', flush=True)


if __name__ == '__main__':
    main()
