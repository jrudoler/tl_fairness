"""Download the Adult-Income dataset to data/raw/ (provenance-capturing step)."""

import argparse

import pandas as pd

URL = "https://raw.githubusercontent.com/socialfoundations/folktables/main/adult_reconstruction.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default=URL)
    parser.add_argument('--output', default='data/raw/adult.csv')
    args = parser.parse_args()
    print(f'Downloading Adult-Income from {args.url}', flush=True)
    df = pd.read_csv(args.url)
    df.to_csv(args.output, index=False)
    print(f'Wrote {args.output} ({df.shape[0]} rows, {df.shape[1]} cols)', flush=True)


if __name__ == '__main__':
    main()
