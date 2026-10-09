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
    ('parity', 'Parity'),
    ('prob_parity', 'Prob. Parity'),
    ('prob_parity_tmle', 'Prob. Parity (TMLE)'),
    ('opportunity', 'Eq. Opp.'),
    ('prob_opp', 'Prob. Eq. Opp.'),
    ('prob_opp_tmle', 'Prob. Eq. Opp. (TMLE)'),
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
    return f"{est:.2f} ({ci[0]:.2f}, {ci[1]:.2f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adult-input', default='data/generated/adult_results.pkl')
    parser.add_argument('--law-input', default='data/generated/law_results.pkl')
    parser.add_argument('--csv-output', default='results/data/real_data_inference.csv')
    parser.add_argument('--tex-output', default='results/data/real_data_inference.tex')
    parser.add_argument('--manuscript-only', action='store_true',
                        help='Omit TMLE comparisons not used in the manuscript')
    parser.add_argument('--body-only', action='store_true')
    args = parser.parse_args()

    adult = _load(args.adult_input)['inference']
    law = _load(args.law_input)['inference']

    rows = [
        {'Metric': label, 'Adult': _fmt(adult, key), 'Law': _fmt(law, key)}
        for key, label in METRIC_LABELS
        if not args.manuscript_only or not key.endswith('_tmle')
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_output, index=False)
    if args.body_only:
        lines=[r'\begin{tabular}{| l || c | c |}',r'\hline',
               r'Metric & Adult & Law \\',r'\hline']
        for row in rows:
            lines.append(f"{row['Metric']} & {row['Adult']} & {row['Law']}"+r' \\')
            if row['Metric'] in ['Prob. Parity','Prob. Eq. Opp.','CMI']:
                lines.append(r'\hline')
        lines.append(r'\end{tabular}')
        Path(args.tex_output).write_text('\n'.join(lines)+'\n')
    else:
        df.to_latex(args.tex_output, index=False)
    print(f'Wrote {args.csv_output} and {args.tex_output}', flush=True)


if __name__ == '__main__':
    main()
