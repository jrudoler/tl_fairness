"""Figure 3: TL variance vs naive difference-in-means variance (Section 4.1.1).

The TL variance estimate is larger than the naive t-test variance because TL
accounts for estimating the conditional distribution, not just the mean.
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
    parser.add_argument('--input', default='data/generated/parity_sim.csv')
    parser.add_argument('--output', default='results/figures/fig3_variance.pdf')
    args = parser.parse_args()

    configure_matplotlib()
    df = pd.read_csv(args.input).sort_values('sample_size')

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))
    ax.plot(df['sample_size'], df['tl_var'], marker='o', label='TL')
    ax.plot(df['sample_size'], df['naive_var'], marker='o', linestyle='--', label='Naive')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Variance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
