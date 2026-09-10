"""Repeated-training audit of probabilistic parity, with exact finite-support truth.

Each outer replicate draws a new training dataset from one fixed population.
Inner replicates draw independent evaluation datasets, keeping that fitted model
fixed. Monte Carlo standard errors cluster by training replicate. The learner
is a saturated four-cell regression with Jeffreys (1/2, 1/2) smoothing; there is
no approximation error, tuning, or randomized optimizer to confound training
sampling noise. Both outcome and propensity are learned on the training data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tlfair.metrics import _prob_group_contrast
from tlfair.plotting import configure_matplotlib

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
METHODS = ("Model fairness", "TL data fairness", "Oracle score")
CELLS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])


def population(effect: float, association: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """X=(U,V) uniform on {-1,1}²; Y and G independent conditional on X.

    D=expit(V + effect*U), pi=(1 + association*U)/2. Under effect=0,
    parity is exactly zero although U predicts G and V predicts Y. This null
    has a nonzero EIF variance. association=0 is an independence control:
    every fixed rule has population parity zero, including fitted rules.
    """
    # Local import avoids cloudpickling a SciPy ufunc from __main__ when the
    # standalone CLI launches loky workers (module-imported tests do not do this).
    from scipy.special import expit

    if not np.isfinite(effect) or not 0 <= association < 1:
        raise ValueError("Require finite effect and 0 <= association < 1")
    return expit(CELLS[:, 1] + effect * CELLS[:, 0]), (1 + association * CELLS[:, 0]) / 2


def exact_gap(scores: np.ndarray, propensity: np.ndarray) -> float:
    """Enumerate E[score(X)|G=1] - E[score(X)|G=0], without MC truth noise."""
    return float(np.sum(scores * propensity) / np.sum(propensity)
                 - np.sum(scores * (1 - propensity)) / np.sum(1 - propensity))


def draw(n: int, rng: np.random.Generator, d: np.ndarray,
         pi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = rng.integers(0, 4, size=n)
    g = rng.binomial(1, pi[cells])
    y = rng.binomial(1, d[cells])
    return cells, g, y


def fit_cells(cells: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Saturated regression, weakly smoothed including any unobserved cells."""
    return ((np.bincount(cells, weights=labels, minlength=4) + 0.5)
            / (np.bincount(cells, minlength=4) + 1))


def model_interval(scores: np.ndarray, group: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Conditional-on-training Welch/normal interval on independent test data."""
    s0, s1 = scores[group == 0], scores[group == 1]
    if min(len(s0), len(s1)) < 2:
        raise ValueError("Need at least two evaluation observations in each group")
    est = float(s1.mean() - s0.mean())
    se = np.sqrt(s0.var(ddof=1) / len(s0) + s1.var(ddof=1) / len(s1))
    return est, (float(est - 1.96 * se), float(est + 1.96 * se))


def one_training(effect: float, association: float, n_train: int, n_test: int,
                 n_evaluations: int, replicate: int, seed: np.random.SeedSequence) -> list[dict]:
    d, pi = population(effect, association)
    truth = exact_gap(d, pi)
    streams = seed.spawn(n_evaluations + 1)
    cells, g, y = draw(n_train, np.random.default_rng(streams[0]), d, pi)
    d_hat, pi_hat = fit_cells(cells, y), fit_cells(cells, g)
    model_truth = exact_gap(d_hat, pi)
    rows = []
    for evaluation, stream in enumerate(streams[1:]):
        cells, g, y = draw(n_test, np.random.default_rng(stream), d, pi)
        predictions = d_hat[cells]
        fits = {
            METHODS[0]: model_interval(predictions, g),
            METHODS[1]: _prob_group_contrast(
                predictions, y - predictions,
                np.column_stack([1 - pi_hat[cells], pi_hat[cells]]),
                np.column_stack([g == 0, g == 1]),
            ),
            METHODS[2]: model_interval(d[cells], g),
        }
        rejects = {method: lo > 0 or hi < 0 for method, (_, (lo, hi)) in fits.items()}
        for method, (est, (lo, hi)) in fits.items():
            own_truth = model_truth if method == METHODS[0] else truth
            rows.append(dict(
                effect=effect, association=association, train_size=n_train,
                test_size=n_test, replicate=replicate, evaluation=evaluation,
                method=method, truth=truth, model_truth=model_truth,
                estimate=est, ci_low=lo, ci_high=hi, se=(hi - lo) / 3.92,
                width=hi - lo, error=est - truth,
                covers_data=lo <= truth <= hi,
                covers_own_target=lo <= own_truth <= hi,
                rejects_zero=rejects[method],
                tl_only_rejects=rejects[METHODS[1]] and not rejects[METHODS[0]],
                model_only_rejects=rejects[METHODS[0]] and not rejects[METHODS[1]],
            ))
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Cluster all MC errors by independently drawn training dataset.

    Nested ANOVA: within = mean conditional empirical variance; between =
    Var(inner means) - within/K. Retain negative estimates of between (finite
    Monte Carlo error), rather than silently clipping a variance component.
    """
    keys = ["effect", "association", "train_size", "test_size", "method"]
    rows = []
    for key, sub in raw.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        clustered = sub.groupby("replicate")
        r = clustered.ngroups
        k = int(clustered.size().iloc[0])
        row.update(truth=float(sub.truth.iloc[0]), training_reps=r, evaluation_reps=k)
        measures = {
            "coverage_data": "covers_data", "coverage_own_target": "covers_own_target",
            "rejection_rate": "rejects_zero", "bias": "error", "mean_ci_width": "width",
            "tl_only_rejects": "tl_only_rejects", "model_only_rejects": "model_only_rejects",
        }
        for name, column in measures.items():
            means = clustered[column].mean()
            row[name] = float(means.mean())
            row[name + "_mcse"] = float(means.std(ddof=1) / np.sqrt(r))
        is_null = abs(row["truth"]) < 1e-12
        row["type1_error"] = row["rejection_rate"] if is_null else np.nan
        row["power"] = row["rejection_rate"] if not is_null else np.nan
        row["false_negative_rate"] = 1 - row["rejection_rate"] if not is_null else np.nan
        row["mean_reported_variance"] = float(np.mean(sub.se ** 2))
        row["training_model_truth_variance"] = float(clustered.model_truth.first().var(ddof=1))
        row["model_truth_bias"] = float(clustered.model_truth.first().mean() - row["truth"])
        within = float(clustered.estimate.var(ddof=1).mean())
        between = float(clustered.estimate.mean().var(ddof=1) - within / k)
        row.update(within_training_variance=within, between_training_variance=between,
                   total_variance=within + between)
        row["empirical_sd"] = float(np.sqrt(max(within + between, 0)))
        rows.append(row)
    return pd.DataFrame(rows)


def run(effects: list[float], train_sizes: list[int], test_sizes: list[int], reps: int,
        evaluations: int, seed: int, n_jobs: int,
        association: float = 0.6, output_dir: Path | None = None,
        ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Sweep both sample sizes; optionally stream raw data one configuration at a time."""
    if (reps < 2 or evaluations < 2 or not test_sizes or min(test_sizes) < 20
            or not train_sizes or min(train_sizes) < 20):
        raise ValueError("Need reps/evaluations >= 2 and train/test sizes >= 20")
    if not effects or any(len(set(values)) != len(values)
                          for values in (effects, train_sizes, test_sizes)):
        raise ValueError("Effects and sample sizes must be nonempty and unique")
    frames, summaries = [], []
    configs = np.random.SeedSequence(seed).spawn(len(effects) * len(train_sizes) * len(test_sizes))
    index = 0
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for effect in effects:
        population(effect, association)
        for n_train in train_sizes:
            for n_test in test_sizes:
                seeds = configs[index].spawn(reps)
                blocks = Parallel(n_jobs=n_jobs)(
                    delayed(one_training)(effect, association, n_train, n_test,
                                          evaluations, rep, stream)
                    for rep, stream in enumerate(seeds)
                )
                frame = pd.DataFrame(row for block in blocks for row in block)
                del blocks
                summaries.append(summarize(frame))
                if output_dir is None:
                    frames.append(frame)
                else:
                    frame.to_csv(output_dir / "replicates.csv.gz", index=False,
                                 mode="w" if index == 0 else "a", header=index == 0,
                                 compression={"method": "gzip", "compresslevel": 1})
                    # Persist partial summaries for monitoring long runs.
                    pd.concat(summaries, ignore_index=True).to_csv(
                        output_dir / "summary.csv", index=False)
                index += 1
                LOG.info("Completed effect=%g, n_train=%d, n_eval=%d: %d fits x %d evaluations",
                         effect, n_train, n_test, reps, evaluations)
    raw = pd.concat(frames, ignore_index=True) if frames else None
    return raw, pd.concat(summaries, ignore_index=True)


def plot(summary: pd.DataFrame, output: Path) -> None:
    """Coverage and rejection share a two-row grid, legends, and rate scale.

    The second PDF page retains interval widths for supplementary inspection.
    Error bars use the original Monte Carlo standard errors without rescaling.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    from tlfair.plotting import FULL_WIDTH

    configure_matplotlib()
    output = Path(output)
    effects = sorted(summary.effect.unique())
    test_sizes = sorted(summary.test_size.unique())
    cmap = LinearSegmentedColormap.from_list(
        "evaluation_size", plt.get_cmap("viridis")(np.linspace(0.08, 0.82, 256)))
    norm = LogNorm(vmin=test_sizes[0], vmax=max(test_sizes[-1], test_sizes[0] * 1.01))
    colors = {n: cmap(norm(n)) for n in test_sizes}
    styles = {"Model fairness": ("--", "s"), "TL data fairness": ("-", "o")}
    with PdfPages(output) as pdf:
        for metrics, suffix in [(["coverage_data", "rejection_rate"], "coverage_rejection"),
                                (["mean_ci_width"], "mean_ci_width")]:
            combined = len(metrics) == 2
            fig, axes = plt.subplots(len(metrics), len(effects), sharey="row",
                                     sharex="col", squeeze=False,
                                     figsize=(FULL_WIDTH, 4.1 if combined else 2.7))
            for row, metric in enumerate(metrics):
                for col, effect in enumerate(effects):
                    ax = axes[row, col]
                    sub = summary[summary.effect == effect]
                    truth = sub.truth.iloc[0]
                    for n_test in test_sizes:
                        for method, (linestyle, marker) in styles.items():
                            points = sub[(sub.test_size == n_test) & (sub.method == method)]
                            points = points.sort_values("train_size")
                            x = points.train_size.to_numpy()
                            y = points[metric].to_numpy()
                            half_width = 1.96 * points[metric + "_mcse"].to_numpy()
                            ax.fill_between(x, y - half_width, y + half_width,
                                            color=colors[n_test], alpha=0.09, linewidth=0)
                            ax.errorbar(x, y, yerr=half_width, color=colors[n_test],
                                        linestyle=linestyle, marker=marker, markersize=2,
                                        markerfacecolor="white", markeredgewidth=0.8,
                                        linewidth=1.2, capsize=2.5, capthick=0.8,
                                        elinewidth=0.8, barsabove=True, zorder=3)
                    ax.set_xscale("log")
                    ax.set_xlim(sub.train_size.min() / 1.2, sub.train_size.max() * 1.2)
                    ticks = [n for n in [100, 500, 2000]
                             if sub.train_size.min() <= n <= sub.train_size.max()]
                    ax.xaxis.set_major_locator(FixedLocator(ticks or sorted(sub.train_size.unique())))
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.tick_params(labelsize=8)
                    ax.minorticks_off()
                    if row == 0:
                        title = "No disparity (null)" if abs(truth) < 1e-12 else f"Disparity: {truth:.3f}"
                        ax.set_title(title, loc="left", pad=5, fontsize=9)
                    if metric != "mean_ci_width":
                        ax.set_ylim(-0.02, 1.03)
                        ax.set_yticks([0, 0.5, 1])
                        if metric == "coverage_data" or abs(truth) < 1e-12:
                            target = 0.95 if metric == "coverage_data" else 0.05
                            ax.axhline(target, color="0.45", linestyle=":", linewidth=1, zorder=1)
                ylabel = {"coverage_data": "Coverage", "rejection_rate": "Rejection rate",
                          "mean_ci_width": "Interval width"}[metric]
                axes[row, 0].set_ylabel(ylabel, fontsize=9)
            fig.supxlabel("Training sample size", fontsize=10, y=0.025)
            method_handles = [Line2D([], [], color="0.2", linestyle=style, marker=marker,
                                     markerfacecolor="white", markersize=4, linewidth=1.6, label=method)
                              for method, (style, marker) in styles.items()]
            fig.legend(handles=method_handles, ncol=2, loc="upper center",
                       bbox_to_anchor=(0.55, 1.01), frameon=False, fontsize=9)
            fig.subplots_adjust(left=0.10, right=0.985, bottom=0.13 if combined else 0.20,
                                top=0.78 if combined else 0.66, wspace=0.17, hspace=0.18)
            color_y = 0.89 if combined else 0.825
            color_ax = fig.add_axes([0.31, color_y, 0.59, 0.018 if combined else 0.027])
            colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax,
                                   orientation="horizontal", ticks=test_sizes)
            colorbar.ax.set_xticklabels([f"{n:,}" for n in test_sizes])
            colorbar.ax.tick_params(labelsize=8, length=2)
            colorbar.ax.minorticks_off()
            colorbar.outline.set_visible(False)
            fig.text(0.29, color_y + 0.009, "Evaluation size", ha="right", va="center", fontsize=9)
            pdf.savefig(fig)
            fig.savefig(output.with_name(output.stem + "_" + suffix + ".png"), dpi=180)
            plt.close(fig)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", nargs="+", type=float, default=[0, 0.2, 0.6])
    parser.add_argument("--association", type=float, default=0.6)
    parser.add_argument("--train-sizes", nargs="+", type=int,
                        default=[100, 150, 225, 350, 500, 750, 1000, 1500, 2000])
    parser.add_argument("--test-sizes", "--test-size", nargs="+", type=int,
                        default=[250, 500, 1000, 2000, 4000],
                        help="evaluation sample sizes swept independently of training sizes")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--evaluations", type=int, default=50,
                        help="independent evaluation datasets per fitted training dataset")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", default="experiments/out/retraining")
    parser.add_argument("--figure", help="optional separate figure path, relative to project root")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw figures from output-dir/summary.csv without rerunning simulations")
    args = parser.parse_args()
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    if args.plot_only:
        summary = pd.read_csv(output / "summary.csv")
        figure = project_path(args.figure) if args.figure else output / "retraining.pdf"
        figure.parent.mkdir(parents=True, exist_ok=True)
        plot(summary, figure)
        LOG.info("Redrew figures from %s", output / "summary.csv")
        return
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    _, summary = run(args.effects, args.train_sizes, args.test_sizes, args.reps,
                     args.evaluations, args.seed, args.n_jobs, args.association, output)
    figure = project_path(args.figure) if args.figure else output / "retraining.pdf"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot(summary, figure)
    LOG.info("Results in %s\n%s", output, summary[["effect", "train_size", "test_size", "method",
             "coverage_data", "coverage_own_target", "rejection_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
