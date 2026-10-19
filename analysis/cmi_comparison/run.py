"""Render paired CMI estimator comparisons and empirical null calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tlfair.plotting import configure_matplotlib, FULL_WIDTH

METHODS = [('one_step','One-step'),('tmle','TMLE')]


def null_diagnostics(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    rows = []
    for (size,method),part in frame[frame.estimator.isin(dict(METHODS))].groupby(['sample_size','estimator']):
        count = len(part)
        covered = ((part.ci_low <= 0)&(part.ci_high >= 0)).mean()
        rejected = (part.estimate-1.645*part.se > 0).mean()
        rows.append({'experiment':experiment,'sample_size':size,'estimator':method,'replicates':count,
                     'bias':part.estimate.mean(),'bias_mcse':part.estimate.std(ddof=1)/np.sqrt(count),
                     'coverage':covered,'coverage_mcse':np.sqrt(covered*(1-covered)/count),
                     'rejection':rejected,'rejection_mcse':np.sqrt(rejected*(1-rejected)/count),
                     'targeting_failures':int((~part.converged.astype(bool)).sum())})
    return pd.DataFrame(rows)


def render(output: Path, table: Path, csv: Path) -> None:
    registry = json.loads((ROOT/'config/publication-runs.json').read_text())['analyses']
    specifications = json.loads((ROOT/'config/analyses.json').read_text())['analyses']
    estimation = ROOT/registry['cmi_estimation']['run']
    conditioning = ROOT/registry['cmi_conditioning_set']['run']
    records = pd.read_csv(estimation/'coverage_replicates.csv.gz')
    conditional = pd.read_csv(conditioning/'replicates.csv.gz')
    included = specifications['cmi_conditioning_set']['parameters']['n-confounders']
    diagnostics = pd.concat([null_diagnostics(records[records.kappa == 0],'Coupling'),
                             null_diagnostics(conditional[conditional.n_confounders_included == included],'Conditioning')],ignore_index=True)
    for path in [output,table,csv]:
        path.parent.mkdir(parents=True,exist_ok=True)
    diagnostics.to_csv(csv,index=False)
    configure_matplotlib()
    coverage = pd.read_csv(ROOT/'data/generated/cmi_estimation_coverage.csv')
    condition = pd.read_csv(ROOT/'data/generated/cmi_conditioning_set.csv')
    fig,axes = plt.subplots(2,2,figsize=(FULL_WIDTH,FULL_WIDTH*.75),layout='constrained')
    for method,label in METHODS:
        curve = coverage[(coverage.estimator == method)&(coverage.sample_size == coverage.sample_size.max())].sort_values('kappa')
        axes[0,0].plot(curve.kappa,curve.error,marker='o',label=label)
        axes[0,1].plot(curve.kappa,curve.coverage,marker='o',label=label)
        curve = condition[(condition.estimator == method)&(condition.sample_size == condition.sample_size.max())].sort_values('n_confounders_included')
        axes[1,0].plot(curve.n_confounders_included,curve.mean_estimate,marker='o',label=label)
        axes[1,1].plot(curve.n_confounders_included,curve.reject_rate,marker='o',label=label)
    for ax,reference in zip(axes.flat,[0,.95,0,.05]):
        ax.axhline(reference,color='grey',ls='--',lw=.8)
        ax.tick_params(labelsize=8)
    axes[0,0].set(xlabel=r'$\kappa$',ylabel='Bias (nats)')
    axes[0,1].set(xlabel=r'$\kappa$',ylabel='Two-sided coverage',ylim=(-.02,1.02))
    axes[1,0].set(xlabel='Confounders included',ylabel='Mean CMI (nats)')
    axes[1,1].set(xlabel='Confounders included',ylabel='One-sided rejection rate',ylim=(-.02,1.02))
    axes[0,0].legend(fontsize=8)
    fig.savefig(output)
    plt.close(fig)
    lines=[r'\small',r'\begin{tabular}{@{}lrlrrr@{}}',r'\toprule',
           r'Experiment & $n$ & Method & Bias & Coverage & Rejection \\',r'\midrule']
    for _,row in diagnostics.sort_values(['experiment','sample_size','estimator']).iterrows():
        bias=f"{row.bias:.3g} ({row.bias_mcse:.2g})"
        covered=f"{row.coverage:.2f} ({row.coverage_mcse:.2f})"
        rejected=f"{row.rejection:.2f} ({row.rejection_mcse:.2f})"
        lines.append(f"{row.experiment} & {int(row.sample_size)} & {dict(METHODS)[row.estimator]} & {bias} & {covered} & {rejected}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    table.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure-output',type=Path,default=ROOT/'results/figures/cmi_paired_comparison.pdf')
    parser.add_argument('--table-output',type=Path,default=ROOT/'results/data/cmi_null_diagnostics.tex')
    parser.add_argument('--csv-output',type=Path,default=ROOT/'results/data/cmi_null_diagnostics.csv')
    args=parser.parse_args()
    render(args.figure_output,args.table_output,args.csv_output)
