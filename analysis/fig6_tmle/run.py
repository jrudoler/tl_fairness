"""Figure 6: one-step vs TMLE vs CV-TMLE for probabilistic parity (Setting 1).

Two panels sharing the sample-size x-axis: (left) coverage of the 95% CI with a
reference line at 0.95; (right) absolute bias. Demonstrates that TMLE matches the
one-step estimator's first-order behaviour while CV-TMLE tightens coverage at
small samples.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tlfair.plotting import configure_matplotlib, FULL_WIDTH

LABELS = {"one_step": "One-step", "tmle": "TMLE", "cv_tmle": "CV-TMLE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/tmle_coverage.csv")
    parser.add_argument("--output", default="results/figures/fig6_tmle_coverage.pdf")
    args = parser.parse_args()

    configure_matplotlib()
    res = pd.read_csv(args.input)
    res["estimator"] = res["estimator"].map(LABELS).fillna(res["estimator"])
    res["abs_bias"] = res["bias"].abs()

    fig, (ax_cov, ax_bias) = plt.subplots(
        1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.42))

    sns.lineplot(data=res, x="sample_size", y="coverage", hue="estimator",
                 marker="o", ax=ax_cov)
    ax_cov.axhline(0.95, ls="--", color="grey", lw=1)
    ax_cov.set_xscale("log")
    ax_cov.set_xlabel("Sample size")
    ax_cov.set_ylabel("95% CI coverage")
    ax_cov.set_ylim(0, 1.02)
    ax_cov.legend(title=None, fontsize=8)

    sns.lineplot(data=res, x="sample_size", y="abs_bias", hue="estimator",
                 marker="o", ax=ax_bias, legend=False)
    ax_bias.set_xscale("log")
    ax_bias.set_xlabel("Sample size")
    ax_bias.set_ylabel("|bias|")

    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
