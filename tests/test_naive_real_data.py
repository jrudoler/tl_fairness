"""Check fixed-model uncertainty against an independent Welch calculation."""

import numpy as np
import pytest
from scipy.stats import ttest_ind

from analysis.naive_real_data.run import fixed_model_interval


def test_fixed_model_se_agrees_with_welch_for_unequal_strata():
    group0 = np.array([0.1, 0.3, 0.6])
    group1 = np.array([0.2, 0.2, 0.5, 0.9, 0.8])
    # These additional observations are ineligible for equal opportunity.
    scores = np.r_[group0, group1, 0.99, 0.01]
    groups = np.r_[np.zeros(len(group0)), np.ones(len(group1)), 0, 1]
    eligible = np.r_[np.ones(8, dtype=bool), False, False]
    result = fixed_model_interval(scores, groups, eligible)
    welch = ttest_ind(group1, group0, equal_var=False)
    independent_se = (group1.mean() - group0.mean()) / welch.statistic
    assert result['se'] == pytest.approx(independent_se)
    assert (result['estimate'] - result['ci_low']) / independent_se == pytest.approx(1.96)
    assert (result['ci_high'] - result['estimate']) / independent_se == pytest.approx(1.96)
    assert (result['n_group0'], result['n_group1']) == (3, 5)


def test_fixed_model_constant_predictions_have_zero_sampling_uncertainty():
    result = fixed_model_interval(np.repeat(0.7, 20), np.r_[np.zeros(8), np.ones(12)])
    assert result['estimate'] == pytest.approx(0, abs=1e-15)
    assert result['se'] == pytest.approx(0, abs=1e-15)
    assert result['ci_low'] == pytest.approx(0, abs=1e-15)
    assert result['ci_high'] == pytest.approx(0, abs=1e-15)
