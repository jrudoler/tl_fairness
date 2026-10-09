"""Figure: conditioning set vs. CMI estimates and empirical Wald rejection rates.

Two panels: (left) mean TL estimate of I(Y;G|X), with shading down to the mean
one-sided 95% lower confidence bound; (right) the one-sided
Wald rejection rate. The nominal alpha line does not imply null calibration.
Reuses the experiment module's ``plot`` so the pipeline figure is identical to the
exploratory one.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.cmi_conditioning_set import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/cmi_conditioning_set.csv")
    parser.add_argument("--output",
                        default="results/figures/cmi_conditioning_set.pdf")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot(pd.read_csv(args.input), args.output)


if __name__ == "__main__":
    main()
