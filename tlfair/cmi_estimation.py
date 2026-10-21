"""Paired CMI estimators from one saved joint-classifier fit."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from tlfair.metrics import _encode_joint, _cmi_log_ratio
from tlfair.tmle import _cmi_target, _wald_ci


CMI_LEARNER_PARAMETERS = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                              validation_fraction=0.2, n_iter_no_change=20, tol=1e-4)


def make_cmi_learner(random_state: int | None = None) -> GradientBoostingClassifier:
    """Use Figure 2's lower-capacity protocol for the four-class joint law."""
    return GradientBoostingClassifier(**CMI_LEARNER_PARAMETERS, random_state=random_state)


def joint_fit_metadata(model: GradientBoostingClassifier) -> dict[str, Any]:
    """Record the stopping outcome as well as the configured maximum size."""
    return {'class': type(model).__name__, 'parameters': model.get_params(),
            'calibration': None, 'full_training_refit': False,
            'boosting_rounds': int(model.n_estimators_),
            'tree_count': int(model.estimators_.size)}


def fit_joint_probabilities(x_train: Any, x_test: Any, y_train: np.ndarray,
                            group_train: np.ndarray, outcome: Any) -> np.ndarray:
    """Fit the supplied joint classifier directly, without probability calibration.

    The publication learner is ``make_cmi_learner``. Its validation subset is
    drawn only from x_train; the fitted model is retained without a full-data refit.
    """
    model = outcome.fit(x_train, _encode_joint(group_train, y_train))
    np.testing.assert_array_equal(model.classes_, np.arange(4))
    return model.predict_proba(x_test)


def paired_cmi(q: np.ndarray, classes: np.ndarray, *, max_iter: int = 200,
               tol: float = 1e-8) -> dict[str, Any]:
    """Keep the original raw one-step estimate and target the same initial fit."""
    q = np.asarray(q, dtype=float)
    classes = np.asarray(classes, dtype=int)
    log_ratio = _cmi_log_ratio(q, q[:, 0]+q[:, 2], q[:, 1]+q[:, 3],
                               q[:, 0]+q[:, 1], q[:, 2]+q[:, 3], classes)
    if not np.isfinite(log_ratio).all():
        raise ValueError('Nonfinite original CMI log ratios')
    one_step = float(log_ratio.mean())
    estimate, eif, diagnostics = _cmi_target(q, classes, max_iter=max_iter, tol=tol)
    substitution, substitution_eif, _ = _cmi_target(q, classes, fluctuate=False)
    return {'inference': {
        'one_step': (one_step, _wald_ci(one_step, log_ratio-one_step)),
        'substitution': (substitution, _wald_ci(substitution, substitution_eif)),
        'tmle': (estimate, _wald_ci(estimate, eif)),
    }, 'diagnostics': diagnostics}


def replicate_records(result: dict[str, Any], **identity: Any) -> list[dict[str, Any]]:
    diagnostics = result['diagnostics']
    fit = result.get('nuisance_fit', {})
    fitted_size = {key: fit[key] for key in ['boosting_rounds', 'tree_count'] if key in fit}
    if fit:
        fitted_size['model_seed'] = fit['parameters']['random_state']
    return [{**identity, **fitted_size, 'estimator': method, 'estimate': estimate,
             'ci_low': interval[0], 'ci_high': interval[1],
             'se': (interval[1]-interval[0])/(2*1.96),
             'converged': diagnostics['converged'] if method == 'tmle' else True,
             'targeting_score': diagnostics['eif_mean'] if method == 'tmle' else np.nan,
             'iterations': diagnostics['iterations'] if method == 'tmle' else 0,
             'termination': diagnostics['termination'] if method == 'tmle' else 'not_targeted',
             'likelihood_gain': diagnostics['final_loglik']-diagnostics['initial_loglik'] if method == 'tmle' else 0.0}
            for method, (estimate, interval) in result['inference'].items()]


def require_convergence(records: pd.DataFrame) -> None:
    """A finite estimate is insufficient to certify a successful targeted run."""
    targeted = records[records.estimator == 'tmle']
    if targeted.empty or not targeted.converged.all():
        failed = targeted.loc[~targeted.converged.astype(bool)]
        raise RuntimeError(f'CMI targeting did not converge in {len(failed)}/{len(targeted)} replicates; retained records describe failures.')
    if not np.isfinite(targeted[['estimate', 'ci_low', 'ci_high', 'targeting_score']]).all().all():
        raise ValueError('Nonfinite targeted CMI results')
    if (targeted.estimate < -1e-12).any() or (targeted.targeting_score.abs() > 1e-8).any():
        raise ValueError('CMI substitution/targeting identities failed')
