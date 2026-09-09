"""Exact-truth and independent-retraining sampling-design regression checks."""

import numpy as np
import pandas as pd
import pytest

from experiments.exp6_retraining import (
    exact_gap, fit_cells, model_interval, population, run,
)


@pytest.mark.parametrize("association", [0.0, 0.6, 0.9])
def test_exact_null_and_alternative(association: float) -> None:
    d, pi = population(0, association)
    assert exact_gap(d, pi) == pytest.approx(0, abs=1e-15)
    d_alt, _ = population(0.6, association)
    assert exact_gap(d_alt, pi) == pytest.approx(association * (d_alt[2] - d_alt[0]))
    if association:
        assert exact_gap(d_alt, pi) > 0


def test_independent_group_null_holds_for_any_fitted_rule() -> None:
    _, pi = population(0.6, 0)
    assert exact_gap(np.array([0.1, 0.9, 0.2, 0.65]), pi) == pytest.approx(0)


def test_smoothing_handles_empty_and_single_class_cells() -> None:
    fit = fit_cells(np.array([0, 0, 2]), np.ones(3))
    np.testing.assert_allclose(fit, [2.5 / 3, 0.5, 0.75, 0.5])


def test_nested_sampling_is_reproducible_and_uses_cluster_mcse() -> None:
    args = dict(effects=[0, 0.2], train_sizes=[100], test_sizes=[100, 200],
                reps=8, evaluations=3, seed=5)
    raw, summary = run(**args, n_jobs=1)
    raw_parallel, summary_parallel = run(**args, n_jobs=2)
    pd.testing.assert_frame_equal(raw, raw_parallel)
    pd.testing.assert_frame_equal(summary, summary_parallel)
    models = raw[(raw.effect == 0) & (raw.test_size == 100) & (raw.method == "Model fairness")]
    assert models.groupby("replicate").model_truth.nunique().eq(1).all()
    assert models.groupby("replicate").estimate.nunique().gt(1).all()
    assert models.groupby("replicate").model_truth.first().nunique() > 1
    cluster_means = models.groupby("replicate").rejects_zero.mean()
    result = summary[(summary.effect == 0) & (summary.test_size == 100)
                     & (summary.method == "Model fairness")].iloc[0]
    assert result.rejection_rate_mcse == pytest.approx(cluster_means.std() / np.sqrt(8))
    assert result.type1_error == pytest.approx(1 - result.coverage_data)


def test_streamed_grid_matches_in_memory_results(tmp_path) -> None:
    args = dict(effects=[0, 0.2], train_sizes=[100, 200], test_sizes=[100, 200],
                reps=3, evaluations=2, seed=71, n_jobs=1)
    raw, summary = run(**args)
    streamed_raw, streamed_summary = run(**args, output_dir=tmp_path)
    assert streamed_raw is None
    pd.testing.assert_frame_equal(summary, streamed_summary)
    pd.testing.assert_frame_equal(raw, pd.read_csv(tmp_path / "replicates.csv.gz"), check_dtype=False)
    assert len(summary) == 2 * 2 * 2 * 3


def test_parallel_cli_from_another_directory(tmp_path) -> None:
    """Exercise __main__ serialization as well as importable-module joblib calls."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "experiments/exp6_retraining.py"
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"))
    subprocess.run([sys.executable, str(script), "--effects", "0", "--train-sizes", "100",
                    "--test-sizes", "100", "200", "--reps", "3", "--evaluations", "2",
                    "--n-jobs", "2", "--output-dir", str(tmp_path / "results")],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    result = pd.read_csv(tmp_path / "results/summary.csv")
    assert set(result.test_size) == {100, 200}
    assert (tmp_path / "results/retraining.pdf").is_file()


def test_oracle_score_sampling_coverage() -> None:
    """A broad deterministic statistical check of conditional CLT calibration."""
    from experiments.exp6_retraining import draw

    rng = np.random.default_rng(831)
    d, pi = population(0.2)
    truth = exact_gap(d, pi)
    covered = []
    for _ in range(500):
        cells, g, _ = draw(500, rng, d, pi)
        _, (lo, hi) = model_interval(d[cells], g)
        covered.append(lo <= truth <= hi)
    assert 0.91 < np.mean(covered) < 0.98
