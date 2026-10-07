"""Render appendix tables from verified, replicate-derived simulation summaries."""
import argparse
from pathlib import Path
import pandas as pd

LABELS = {'default': 'Boost', 'higher_capacity': 'High', 'linear': 'Linear',
          'quadratic': 'Quadratic', 'oracle': 'Oracle', 'cross_validated': 'CV'}
SCENARIOS = {'primary': 'Primary ($\\lambda=0.5$)', 'linear': 'Linear ($\\lambda=0$)',
             'nonlinear': 'More nonlinear ($\\lambda=0.75$)', 'stress': 'Squared-feature stress test'}
FITTED = ['default', 'higher_capacity', 'boost_linear', 'linear_boost', 'linear_linear']
CONTROLS = ['oracle_oracle', 'oracle_linear', 'linear_oracle', 'quadratic_quadratic',
            'quadratic_linear', 'linear_quadratic']


def write_table(df, path, caption, label, sections):
    lines = [r'{\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{longtable}{rllrrrr}',
             '\\caption{'+caption+'}\\label{'+label+r'}\\',
             r'\toprule', r'$n_{\rm train}$ & $(\hat D,\hat\pi)$ & Method & Coverage & Bias (pp) & SD (pp) & SE (pp)\\',
             r'\midrule', r'\endfirsthead',
             r'\multicolumn{7}{l}{\tablename\ \thetable\ continued}\\',
             r'\toprule', r'$n_{\rm train}$ & $(\hat D,\hat\pi)$ & Method & Coverage & Bias (pp) & SD (pp) & SE (pp)\\',
             r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
    for title, part, cases in sections:
        lines += [r'\multicolumn{7}{l}{\textit{'+title+r'}}\\*']
        for n in sorted(part.train_size.unique()):
            for case in cases:
                sub = part[(part.train_size == n) & (part.case == case)]
                for method in ['TL data fairness', 'Model fairness']:
                    rows = sub[sub.method == method]
                    if len(rows) != 1:
                        raise ValueError(f'Expected one summary: {title}/{n}/{case}/{method}')
                    row = rows.iloc[0]
                    spec = f'({LABELS[row.outcome_model]}, {LABELS[row.group_model]})'
                    name = 'TL' if method.startswith('TL') else 'Model'
                    lines.append(f'{n:,} & {spec} & {name} & '
                                 f'{row.coverage:.3f} ({row.coverage_mcse:.3f}) & '
                                 f'{100*row.bias:.2f} ({100*row.bias_mcse:.3f}) & '
                                 f'{100*row.empirical_sd:.2f} & {100*row.mean_se:.2f}'+(r'\\*' if name == 'TL' else r'\\'))
            lines.append(r'\addlinespace')
    # A trailing spacing row can force an otherwise empty continuation page.
    if lines[-1] == r'\addlinespace':
        lines.pop()
    lines += [r'\end{longtable}', '}']
    path.write_text('\n'.join(lines)+'\n')


def findings(summary, paired):
    """Numerical statements are generated from full-run aggregates and paired MCSEs."""
    def row(case, n, method):
        return summary[(summary.scenario == 'primary') & (summary.case == case)
                       & (summary.train_size == n) & (summary.method == method)].iloc[0]
    paragraphs = []
    cases = [('default', 'lower-capacity'), ('higher_capacity', 'higher-capacity')]
    if 'cross_validated' in set(summary.case):
        cases.insert(1, ('cross_validated', 'cross-validated'))
    for case, label in cases:
        tl, model = row(case, 1000, 'TL data fairness'), row(case, 1000, 'Model fairness')
        comparison = paired[(paired.scenario == 'primary') & (paired.case == case)
                            & (paired.train_size == 1000)].iloc[0]
        diff, se = 100*comparison.coverage_difference, 100*comparison.coverage_difference_mcse
        paragraphs.append(
            f'At training size 1000 with {label} boosting, model-fairness and TL coverage are '
            f'${model.coverage:.3f}$ and ${tl.coverage:.3f}$, respectively. '
            f'The paired TL-minus-model coverage difference is ${diff:.1f}$ percentage points '
            f'(approximate Monte Carlo interval $[{diff-1.96*se:.1f}, {diff+1.96*se:.1f}]$). '
            f'Their signed biases are ${100*model.bias:.2f}$ and ${100*tl.bias:.2f}$ percentage points, respectively.')
    tl, model = row('higher_capacity', 10000, 'TL data fairness'), row('higher_capacity', 10000, 'Model fairness')
    comparison = paired[(paired.scenario == 'primary') & (paired.case == 'higher_capacity')
                        & (paired.train_size == 10000)].iloc[0]
    diff, se = 100*comparison.coverage_difference, 100*comparison.coverage_difference_mcse
    resolution = ('This coverage difference is not clearly resolved by Monte Carlo uncertainty. '
                  if diff-1.96*se <= 0 <= diff+1.96*se else '')
    paragraphs.append(
        f'With 10000 training observations and higher-capacity boosting, coverage is '
        f'${model.coverage:.3f}$ for model fairness and ${tl.coverage:.3f}$ for TL '
        f'(paired difference ${diff:.1f}$ percentage points, Monte Carlo interval '
        f'$[{diff-1.96*se:.1f}, {diff+1.96*se:.1f}]$). '
        f'The corresponding signed biases are ${100*model.bias:.2f}$ and ${100*tl.bias:.2f}$ percentage points. '
        + resolution +
        'The size of the discrepancy therefore depends on learner accuracy and interval precision; '
        'these comparisons do not establish uniform superiority of TL.')
    return '\n\n'.join(paragraphs)+'\n'


def render(summary, population, output, paired):
    if not {"reps", "test_size"}.issubset(summary.columns) or not summary.reps.eq(1000).all() or not summary.test_size.eq(2000).all():
        raise ValueError("Publication tables require 1000 replicates and evaluation size 2000")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = summary[summary.train_size.isin([500, 2500, 10000])]
    primary = selected[selected.scenario == 'primary']
    note = ('All intervals are evaluated against population data fairness, with evaluation size 2000 '
            'and 1000 independent replicates per setting. Parentheses contain Monte Carlo standard errors '
            'for coverage and signed bias. Bias, empirical estimator standard deviation (SD), and mean '
            'reported standard error (SE) are in percentage points (pp). An empirical coverage of zero or '
            'one has zero plug-in Monte Carlo standard error; this does not establish an exact coverage probability. ')
    write_table(primary, output/'nuisance_fitted.tex', note + 'Estimated nuisance comparisons; Boost and High '
                'denote lower- and higher-capacity gradient-boosted decision-tree classifiers. Model uses the same outcome predictions as TL '
                'and does not use the group fit.', 'tab:nuisance-fitted',
                [('Primary family', primary, FITTED[:1]+(['cross_validated'] if 'cross_validated' in set(primary.case) else [])+FITTED[1:])])
    write_table(primary, output/'nuisance_controls.tex', note + 'Oracle probabilities and correctly specified '
                'quadratic logistic fits are distinguished explicitly. Linear fits use raw features; Quadratic '
                'fits use raw and squared features.', 'tab:nuisance-controls',
                [('Specification controls', primary, CONTROLS)])
    write_table(selected, output/'nuisance_sensitivity.tex', note + 'Matched learner comparisons under '
                'alternative nonlinearity levels and the original squared-feature stress test.',
                'tab:nuisance-sensitivity',
                [(SCENARIOS[s], selected[selected.scenario == s], ['default', 'higher_capacity'])
                 for s in ['linear', 'nonlinear', 'stress']])
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Population characteristics, calculated by replicated scrambled-Sobol integration. '
             r'Group prevalence and the 5th and 95th propensity percentiles describe the group distribution. '
             r'The disparity is expressed in percentage points; its numerical integration standard error is reported separately.}',
             r'\label{tab:nuisance-population}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Setting & $\Pr(G=1)$ & $\pi_{.05}$ & $\pi_{.95}$ & Gap (pp) & Integration SE (pp)\\', r'\midrule']
    for scenario in SCENARIOS:
        row = population[population.scenario == scenario].iloc[0]
        lines.append(f'{SCENARIOS[scenario]} & {row.group_prevalence:.3f} & {row.propensity_q05:.3f} & '
                     f'{row.propensity_q95:.3f} & {100*row.truth:.2f} & {100*row.integration_se:.5f}'+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (output/'nuisance_population.tex').write_text('\n'.join(lines)+'\n')
    selected.to_csv(output/'nuisance_tables.csv', index=False)
    (output/'nuisance_findings.tex').write_text(findings(summary, paired))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/generated/nuisance_early_stopping/summary.csv')
    p.add_argument('--population', default='data/generated/nuisance_early_stopping/population.csv')
    p.add_argument('--paired', default='data/generated/nuisance_early_stopping/paired_comparisons.csv')
    p.add_argument('--output-dir', default='results/data')
    a = p.parse_args()
    render(pd.read_csv(a.input), pd.read_csv(a.population), a.output_dir, pd.read_csv(a.paired))
