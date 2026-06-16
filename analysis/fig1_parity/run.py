"""Figure 1: demographic parity estimates vs sample size (paper Section 4.1).

Top panel: per-replicate-mean estimate vs sample size with the true value.
Bottom panel: mean estimate with a 95% Wald CI band, on a log sample-size axis.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tlfair.plotting import configure_matplotlib, FULL_WIDTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/generated/parity_sim.csv')
    parser.add_argument('--output', default='results/figures/fig1_parity.pdf')
    args = parser.parse_args()

    configure_matplotlib()
    df = pd.read_csv(args.input).sort_values('sample_size')
    truth = df['truth'].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(FULL_WIDTH, 5), sharex=True)

    axes[0].scatter(df['sample_size'], df['estimate'], s=25)
    axes[0].axhline(truth, color='black', linestyle='--', label='truth')
    axes[0].set_xscale('log')
    axes[0].set_ylabel('Estimate')
    axes[0].legend()

    axes[1].errorbar(df['sample_size'], df['estimate'],
                     yerr=1.96 * df['std'], fmt='o', capsize=3)
    axes[1].axhline(truth, color='black', linestyle='--')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Sample Size')
    axes[1].set_ylabel('Estimate (95% CI)')

    fig.tight_layout()
    fig.savefig(args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
