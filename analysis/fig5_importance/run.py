"""Figure 5: per-metric feature importance for Adult (top) and Law (bottom).

A 2x5 grid of horizontal bar plots, one column per fairness metric (paper
Section 5, Figure 5).
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

TITLES = ['Parity', 'Prob. Parity', 'Opportunity', 'Prob. Opportunity', 'CMI']


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adult-input', default='data/generated/adult_results.pkl')
    parser.add_argument('--law-input', default='data/generated/law_results.pkl')
    parser.add_argument('--output', default='results/figures/fig5_importance.pdf')
    args = parser.parse_args()

    configure_matplotlib()
    adult = _load(args.adult_input)
    law = _load(args.law_input)
    metrics = list(law['importance'].keys())

    fig, axs = plt.subplots(2, len(metrics), figsize=(FULL_WIDTH * 2.3, 8))
    for i, m in enumerate(metrics):
        adult_res = adult['importance'][m][0]
        law_res = law['importance'][m][0]
        adult_df = pd.DataFrame({'Variable': list(adult_res.keys()), 'Importance': list(adult_res.values())})
        law_df = pd.DataFrame({'Variable': list(law_res.keys()), 'Importance': list(law_res.values())})

        sns.barplot(adult_df, y='Variable', x='Importance', ax=axs[0, i])
        sns.barplot(law_df, y='Variable', x='Importance', ax=axs[1, i])
        axs[0, i].set_title(TITLES[i], fontsize=11)
        axs[0, i].set_xlabel('')
        if i >= 1:
            axs[0, i].set_yticklabels([])
            axs[1, i].set_yticklabels([])
            axs[0, i].set_ylabel('')
            axs[1, i].set_ylabel('')

    fig.tight_layout()
    fig.savefig(args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
