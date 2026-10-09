"""Monte Carlo ground-truth CMI for each value of kappa.

Produces the table in the paper's "CMI Simulation" appendix.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/generated/cmi_population_truth.pkl')
    parser.add_argument('--csv-output', default='results/data/cmi_population_truth.csv')
    parser.add_argument('--tex-output', default='results/data/cmi_population_truth.tex')
    parser.add_argument('--body-only', action='store_true')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        truth = pickle.load(f)

    df = pd.DataFrame(
        sorted(truth.items()), columns=['kappa', 'CMI']
    )
    df['CMI'] = df['CMI'].round(4)
    df.to_csv(args.csv_output, index=False)
    if args.body_only:
        lines=[r'\begin{tabular}{c|c}',r'$\kappa$ & $I(Y;G\mid X)$ \\ \hline']
        lines += [f'{k:g} & {v:.4f}'+r' \\' for k,v in sorted(truth.items())]
        lines.append(r'\end{tabular}')
        Path(args.tex_output).write_text('\n'.join(lines)+'\n')
    else:
        df.to_latex(args.tex_output, index=False)
    print(f'Wrote {args.csv_output} and {args.tex_output}', flush=True)


if __name__ == '__main__':
    main()
