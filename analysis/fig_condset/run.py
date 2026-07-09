"""Figure: conditioning set vs. data-fairness verdict (CMI).

Two panels: (left) mean TL estimate of I(Y;G|X) with its mean 95% CI band, falling
to zero as confounders are added to the conditioning set; (right) the one-sided
Wald flag rate, near alpha with all confounders and approaching one without them.
Reuses the experiment module's ``plot`` so the pipeline figure is identical to the
exploratory one.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.exp4_feature_selection import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/condset.csv")
    parser.add_argument("--output",
                        default="results/figures/fig8_conditioning_set.pdf")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot(pd.read_csv(args.input), args.output)


if __name__ == "__main__":
    main()
