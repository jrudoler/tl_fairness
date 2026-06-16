"""Download the Law School dataset to data/raw/ (provenance-capturing step)."""

import argparse

import pandas as pd

URL = "https://raw.githubusercontent.com/tailequy/fairness_dataset/refs/heads/main/experiments/data/law_school_clean.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default=URL)
    parser.add_argument('--output', default='data/raw/law.csv')
    args = parser.parse_args()
    print(f'Downloading Law School from {args.url}', flush=True)
    df = pd.read_csv(args.url)
    df.to_csv(args.output, index=False)
    print(f'Wrote {args.output} ({df.shape[0]} rows, {df.shape[1]} cols)', flush=True)


if __name__ == '__main__':
    main()
