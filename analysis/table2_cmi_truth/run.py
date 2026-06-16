"""Table 2 (Appendix A): Monte Carlo ground-truth CMI for each value of c."""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/generated/truth_dict.pkl')
    parser.add_argument('--csv-output', default='results/data/table2_cmi_truth.csv')
    parser.add_argument('--tex-output', default='results/data/table2_cmi_truth.tex')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        truth = pickle.load(f)

    df = pd.DataFrame(
        sorted(truth.items()), columns=['c', 'CMI']
    )
    df['CMI'] = df['CMI'].round(4)
    df.to_csv(args.csv_output, index=False)
    df.to_latex(args.tex_output, index=False)
    print(f'Wrote {args.csv_output} and {args.tex_output}', flush=True)


if __name__ == '__main__':
    main()
