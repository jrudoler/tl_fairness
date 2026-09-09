"""Figure 2: double-robustness coverage heatmap (paper Section 4.1, Setting 3).

Rows identify the fitted learners, with labels derived from
the case names so they cannot drift out of alignment with the data. (The
published figure had the top/bottom row labels swapped.)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tlfair.plotting import configure_matplotlib, COLUMN_WIDTH

# Historical case keys are retained in CSVs; boosting is not an oracle fit.
CASE_TO_LABEL = {
    'both_correct': 'Both boosting',
    'outcome_correct': 'Only outcome boosting',
    'propensity_correct': 'Only group boosting',
    'misspecified': 'Both linear',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/generated/robust_res.csv')
    parser.add_argument('--output', default='results/figures/fig2_robust_coverage.pdf')
    args = parser.parse_args()

    configure_matplotlib()
    res = pd.read_csv(args.input)
    res['cases'] = pd.Categorical(res['cases'], list(CASE_TO_LABEL.keys()))
    plot = res.pivot(index='cases', columns='sample_size', values='coverage')

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.6, COLUMN_WIDTH))
    sns.heatmap(plot, annot=True, ax=ax, vmin=0, vmax=1,
                cmap=plt.rcParams["image.cmap"])
    ax.set_yticklabels([CASE_TO_LABEL[c] for c in plot.index], rotation=0)
    ax.set_ylabel('Scenario')
    ax.set_xlabel('Sample Size')
    fig.tight_layout()
    fig.savefig(args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
