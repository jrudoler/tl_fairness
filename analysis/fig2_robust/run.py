"""Figure 2: double-robustness coverage heatmap (paper Section 4.1, Setting 3).

Rows are ordered well-specified -> fully misspecified, with labels derived from
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

# case name -> scenario label. both_correct = both plug-ins correct
# (well-specified); outcome_correct = only Y|X correct so G|X is misspecified;
# propensity_correct = only G|X correct so Y|X is misspecified; misspecified =
# neither correct (fully misspecified).
CASE_TO_LABEL = {
    'both_correct': 'Well-Specified',
    'outcome_correct': 'G|X Misspecified',
    'propensity_correct': 'Y|X Misspecified',
    'misspecified': 'Fully Misspecified',
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
    sns.heatmap(plot, annot=True, ax=ax)
    ax.set_yticklabels([CASE_TO_LABEL[c] for c in plot.index], rotation=0)
    ax.set_ylabel('Scenario')
    ax.set_xlabel('Sample Size')
    fig.tight_layout()
    fig.savefig(args.output)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
