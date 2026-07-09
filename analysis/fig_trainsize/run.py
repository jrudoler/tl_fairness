"""Figure: training-size vs. naive-baseline coverage (Setting 3).

Two panels sharing the (log) training-size x-axis: (left) 95% CI coverage with a
reference line at 0.95; (right) bias. Reuses the experiment module's ``plot`` so
the pipeline figure is identical to the exploratory one.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.exp5_trainsize_coverage import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/trainsize_coverage.csv")
    parser.add_argument("--output",
                        default="results/figures/fig7_trainsize_coverage.pdf")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot(pd.read_csv(args.input), args.output)


if __name__ == "__main__":
    main()
