"""Table 1: inferential results (estimate + 95% CI) for Adult and Law (Section 5)."""

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
    parser.add_argument('--csv-output', default='results/data/table1_inference.csv')
    parser.add_argument('--tex-output', default='results/data/table1_inference.tex')
    args = parser.parse_args()

    adult = _load(args.adult_input)['inference']
    law = _load(args.law_input)['inference']

    rows = [
        {'Metric': label, 'Adult': _fmt(adult, key), 'Law': _fmt(law, key)}
        for key, label in METRIC_LABELS
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_output, index=False)
    df.to_latex(args.tex_output, index=False)
    print(f'Wrote {args.csv_output} and {args.tex_output}', flush=True)


if __name__ == '__main__':
    main()
