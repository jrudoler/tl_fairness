"""Matched learner comparison: coverage and signed bias with Monte Carlo bars."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from tlfair.plotting import configure_matplotlib, FULL_WIDTH


def plot(df, output):
    required = {'scenario', 'case', 'method', 'train_size', 'coverage', 'coverage_mcse', 'bias', 'bias_mcse', 'reps', 'test_size'}
    if not required.issubset(df.columns):
        raise ValueError('Figure requires replicate-derived summaries, including Monte Carlo errors')
    if not df.reps.eq(1000).all() or not df.test_size.eq(2000).all():
        raise ValueError('Publication figure requires 1000 replicates and evaluation size 2000')
    df = df[(df.scenario == 'primary') & df.case.isin(['default', 'cross_validated', 'higher_capacity'])]
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.15), sharex=True)
    colors = {'default': '#0072B2', 'higher_capacity': '#D55E00'}
    if 'cross_validated' in set(df.case):
        colors = {'default': '#0072B2', 'cross_validated': '#009E73', 'higher_capacity': '#D55E00'}
    styles = {'TL data fairness': ('-', 'o'), 'Model fairness': ('--', 's')}
    for case, color in colors.items():
        for method, (style, marker) in styles.items():
            sub = df[(df.case == case) & (df.method == method)].sort_values('train_size')
            if sub.empty or sub.train_size.duplicated().any():
                raise ValueError(f'Missing or duplicate curve {case}/{method}')
            p = sub.coverage.to_numpy()
            half = 1.96*sub.coverage_mcse.to_numpy()
            for ax, values, errors in [
                (axes[0], p, np.vstack([np.minimum(half, p), np.minimum(half, 1-p)])),
                (axes[1], 100*sub.bias, 196*sub.bias_mcse),
            ]:
                ax.errorbar(sub.train_size, values, yerr=errors, color=color, linestyle=style,
                            marker=marker, markersize=4, linewidth=1.3, elinewidth=.8,
                            capsize=2, markerfacecolor=color if method == 'TL data fairness' else 'white',
                            markeredgewidth=.8)
    axes[0].axhline(.95, color='.45', ls=':', lw=1, zorder=0)
    axes[0].set_ylim(-.02, 1.02)
    axes[1].axhline(0, color='.45', ls=':', lw=1, zorder=0)
    axes[0].set_ylabel('Coverage')
    axes[1].set_ylabel('Bias')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Training sample size')
        ax.tick_params(labelsize=8)
    labels = {'default': 'Default boosting', 'cross_validated': 'CV-tuned boosting',
              'higher_capacity': 'Higher-capacity boosting'}
    handles = [Line2D([], [], color=c, lw=2, label=labels[case])
               for case, c in colors.items()]
    handles += [Line2D([], [], color='.2', linestyle=ls, marker=marker, markersize=4,
                      markerfacecolor='.2' if label == 'TL data fairness' else 'white', label=label)
                for label, (ls, marker) in styles.items()]
    if len(colors) == 3:
        handles = [handles[0], handles[3], handles[1], handles[4], handles[2]]
    fig.legend(handles=handles, loc='lower center', ncol=len(colors), frameon=False, fontsize=8,
               columnspacing=1.7, handlelength=2.8, bbox_to_anchor=(.5, -.005))
    fig.tight_layout(rect=(0, .17, 1, 1), w_pad=1.3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/generated/nuisance_early_stopping/summary.csv')
    p.add_argument('--output', default='results/figures/fig7_trainsize_coverage.pdf')
    args = p.parse_args()
    plot(pd.read_csv(args.input), args.output)
