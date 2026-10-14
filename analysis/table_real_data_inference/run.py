"""Real-data inference: estimates and 95% CIs for Adult and Law."""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

# inference-dict key -> display label, in paper row order. TMLE rows are shown
# beneath their one-step counterparts for the doubly-robust metrics.
METRIC_LABELS = [
    ('prob_parity', 'Prob. Parity'),
    ('prob_parity_tmle', 'Prob. Parity (TMLE)'),
    ('prob_opp', 'Prob. Eq. Opp.'),
    ('prob_opp_tmle', 'Prob. Eq. Opp. (TMLE)'),
    ('parity', 'Parity (thresholded)'),
    ('opportunity', 'Eq. Opp. (thresholded)'),
    ('cmi', 'CMI'),
    ('cmi_tmle', 'CMI (TMLE)'),
]


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _fmt(inference, key):
    if key not in inference:
        return "--"  # metric not run (e.g. a --metrics subset)
    est, ci = inference[key]
    precision = 4 if key in ['cmi','cmi_tmle'] else 3
    return f"{est:.{precision}f} ({ci[0]:.{precision}f}, {ci[1]:.{precision}f})"


def comparison_rows(original: dict, matched: dict, dataset: str, manuscript_only: bool) -> list[dict[str,str]]:
    data = {**original, **matched['data_fairness']}
    return [{'Metric':label,'Dataset':dataset,'Data fairness':_fmt(data,key),
             'Model fairness':_fmt(matched['model_fairness'],key)}
            for key,label in METRIC_LABELS if not manuscript_only or not key.endswith('_tmle')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adult-input', default='data/generated/adult_results.pkl')
    parser.add_argument('--law-input', default='data/generated/law_results.pkl')
    parser.add_argument('--adult-comparison')
    parser.add_argument('--law-comparison')
    parser.add_argument('--csv-output', default='results/data/real_data_inference.csv')
    parser.add_argument('--tex-output', default='results/data/real_data_inference.tex')
    parser.add_argument('--manuscript-only', action='store_true',
                        help='Omit TMLE comparisons not used in the manuscript')
    parser.add_argument('--body-only', action='store_true')
    args = parser.parse_args()

    adult = _load(args.adult_input)['inference']
    law = _load(args.law_input)['inference']

    if bool(args.adult_comparison) != bool(args.law_comparison):
        parser.error('Supply both dataset comparisons together')
    # The existing selected publication remains buildable while new fits queue.
    if not args.adult_comparison:
        rows = [{'Metric':label,'Adult':_fmt(adult,key),'Law':_fmt(law,key)}
                for key,label in METRIC_LABELS if not args.manuscript_only or not key.endswith('_tmle')]
        df = pd.DataFrame(rows)
        df.to_csv(args.csv_output,index=False)
        df.to_latex(args.tex_output,index=False)
        return
    adult_rows = comparison_rows(adult,_load(args.adult_comparison),'Adult',args.manuscript_only)
    law_rows = comparison_rows(law,_load(args.law_comparison),'Law',args.manuscript_only)
    rows = [row for pair in zip(adult_rows,law_rows) for row in pair]
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_output, index=False)
    if args.body_only:
        lines=[r'\small',r'\begin{tabular}{@{}llcc@{}}',r'\toprule',
               r'Metric & Dataset & Data fairness & Model fairness \\',r'\midrule']
        for row in rows:
            label = row['Metric'] if row['Dataset']=='Adult' else ''
            lines.append(f"{label} & {row['Dataset']} & {row['Data fairness']} & {row['Model fairness']}"+r' \\')
            if row['Dataset']=='Law':
                lines.append(r'\midrule' if row['Metric'] in ['Prob. Eq. Opp.','Eq. Opp. (thresholded)'] else r'\addlinespace')
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        Path(args.tex_output).write_text('\n'.join(lines)+'\n')
    else:
        df.to_latex(args.tex_output, index=False)
    print(f'Wrote {args.csv_output} and {args.tex_output}', flush=True)


if __name__ == '__main__':
    main()
