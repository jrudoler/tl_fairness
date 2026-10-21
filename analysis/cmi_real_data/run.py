"""CMI-only real-data comparison using Figure 2's depth-3 fitting protocol."""
from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tlfair.metrics import _encode_joint
from tlfair.cmi_estimation import fit_joint_probabilities, paired_cmi, replicate_records, require_convergence
from tlfair.cmi_estimation import make_cmi_learner, joint_fit_metadata
from tlfair.provenance import sha256, write_json

LOG = logging.getLogger(__name__)


def run(dataset: str, input_path: Path, output: Path, seed: int,
        max_iter: int = 200, tol: float = 1e-8) -> None:
    loader = importlib.import_module(f'analysis.analyze_{dataset}.run')
    xtr, xte, ytr, yte, gtr, gte = loader.load_data(input_path, seed)
    LOG.info('Fitting depth-3 joint classifier with early stopping: %s, train=%d evaluation=%d', dataset, len(xtr), len(xte))
    outcome = make_cmi_learner(seed)
    q = fit_joint_probabilities(xtr, xte, ytr, gtr, outcome)
    result = paired_cmi(q, _encode_joint(gte, yte), max_iter=max_iter, tol=tol)
    result['nuisance_fit'] = joint_fit_metadata(outcome)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.parent/'cmi_predictions.npz'
    diagnostics = result['diagnostics']
    np.savez_compressed(cache, initial_raw_q=q, initial_q=diagnostics['initial_q'],
                        targeted_q=diagnostics['q_star'], outcomes=yte, groups=gte,
                        train_index=xtr.index.to_numpy(), test_index=xte.index.to_numpy())
    inference = {f'cmi_{method}': value for method, value in result['inference'].items()}
    with output.open('wb') as stream:
        pickle.dump({'inference': inference, 'prediction_sha256': sha256(cache)}, stream)
    scalar_diagnostics = {key:value for key,value in diagnostics.items() if key not in ['q_star','initial_q']}
    write_json(output.parent/'cmi_fit_metadata.json', {
        'dataset':dataset, 'seed':seed, 'train_rows':len(xtr), 'evaluation_rows':len(xte),
        'raw_sha256':sha256(input_path), 'preprocessing_sha256':sha256(Path(loader.__file__)),
        'prediction_sha256':sha256(cache), 'calibration_folds':0,
        'learner':outcome.get_params(), 'nuisance_fit':result['nuisance_fit'],
        'diagnostics':scalar_diagnostics})
    records = pd.DataFrame(replicate_records(result, dataset=dataset))
    records.to_csv(output.parent/'cmi_diagnostics.csv', index=False)
    LOG.info('CMI estimates: %s; targeting: %s', inference, scalar_diagnostics)
    require_convergence(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['adult','law'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args.dataset, args.input, args.output, args.seed, args.max_iter, args.tol)


if __name__ == '__main__':
    main()
