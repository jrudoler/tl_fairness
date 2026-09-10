"""Side-by-side coverage for boosting fits and exact/quadratic controls."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tlfair.plotting import configure_matplotlib, FULL_WIDTH

LOG = logging.getLogger(__name__)
# Each pair is (outcome estimator D_hat, group estimator pi_hat).
BOOSTING_CASES = {
    'both_correct': '(Boost, Boost)',
    'outcome_correct': '(Boost, Linear)',
    'propensity_correct': '(Linear, Boost)',
    'misspecified': '(Linear, Linear)',
}
CONTROL_CASES = {
    'both_true': '(True, True)',
    'true_outcome_linear_group': '(True, Linear)',
    'linear_outcome_true_group': '(Linear, True)',
    'quadratic_outcome_linear_group': '(Quadratic, Linear)',
    'linear_outcome_quadratic_group': '(Linear, Quadratic)',
    'both_quadratic': '(Quadratic, Quadratic)',
}


def plot(results: pd.DataFrame, controls: pd.DataFrame, output: str | Path) -> None:
    """Keep every saved coverage value, with a common color scale and labels."""
    configure_matplotlib()
    boost = results.pivot(index='cases', columns='sample_size', values='coverage').reindex(BOOSTING_CASES)
    exact = controls.pivot(index='case', columns='n_train', values='coverage').reindex(CONTROL_CASES)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6), gridspec_kw={'width_ratios': [1.3, 1]})
    color_ax = fig.add_axes([0.945, 0.27, 0.017, 0.55])
    for ax, values, labels, title, xlabel, fmt in [
        (axes[0], boost, BOOSTING_CASES, '(a) Boosting fits', 'Total sample size', '.2f'),
        (axes[1], exact, CONTROL_CASES, '(b) Controls', 'Training sample size', '.4f'),
    ]:
        sns.heatmap(values, annot=True, fmt=fmt, annot_kws={'size': 8}, ax=ax,
                    vmin=0, vmax=1, cmap=plt.rcParams['image.cmap'],
                    cbar=ax is axes[1], cbar_ax=color_ax if ax is axes[1] else None,
                    cbar_kws={'ticks': [0, 0.5, 1]})
        ax.set_yticklabels(list(labels.values()), rotation=0, fontsize=7.5)
        ax.set_xticklabels([f'{int(n):,}' for n in values.columns], rotation=0, fontsize=8)
        ax.set_ylabel(r'$(\widehat D,\widehat\pi)$', fontsize=9, labelpad=3)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=9, loc='left', pad=8)
    color_ax.set_title('Coverage', fontsize=8, pad=10)
    color_ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.155, right=0.925, bottom=0.20, top=0.82, wspace=1.05)
    fig.savefig(output, bbox_inches='tight')
    fig.savefig(Path(output).with_suffix('.png'), bbox_inches='tight', dpi=180)
    plt.close(fig)
    LOG.info('Wrote %s', output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/generated/robust_res.csv')
    parser.add_argument('--controls-input', default='experiments/out/eif_regeneration/dr_controls/summary.csv')
    parser.add_argument('--output', default='results/figures/fig2_robust_coverage.pdf')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    plot(pd.read_csv(args.input), pd.read_csv(args.controls_input), args.output)


if __name__ == '__main__':
    main()
