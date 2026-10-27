"""Data- and model-fairness intervals from shared held-out predictions."""
from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import pickle
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from tlfair.metrics import _encode_joint, _prob_group_contrast
from tlfair.provenance import sha256, write_json
from tlfair.superlearner import SuperLearnerClassifier

LOG = logging.getLogger(__name__)


def summarize_predictions(probabilities: np.ndarray, decisions: np.ndarray,
                          group_probabilities: np.ndarray, joint_probabilities: np.ndarray,
                          outcomes: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """Use the same outcome fit and variance convention for both targets."""
    residual = (outcomes == 1) - probabilities
    parity_strata = np.column_stack([groups == 0, groups == 1])
    opportunity_strata = parity_strata & (outcomes == 1)[:, None]
    zero = np.zeros(len(outcomes))
    data, model = {}, {}
    for name, scores, strata, weights, correction in [
        ('prob_parity',probabilities,parity_strata,group_probabilities,residual),
        ('prob_opp',probabilities,opportunity_strata,joint_probabilities[:,[2,3]],residual),
        ('parity',decisions,parity_strata,np.zeros_like(parity_strata,dtype=float),zero),
        ('opportunity',decisions,opportunity_strata,np.zeros_like(opportunity_strata,dtype=float),zero),
    ]:
        data[name] = _prob_group_contrast(scores,correction,weights,strata)
        model[name] = _prob_group_contrast(scores,zero,np.zeros_like(strata,dtype=float),strata)
    return {'data_fairness':data,'model_fairness':model}


def fit_predict(role: str, xtr: pd.DataFrame, xte: pd.DataFrame,
                labels: np.ndarray, seed: int, inner_jobs: int) -> dict[str, Any]:
    started = time.perf_counter()
    learner = SuperLearnerClassifier(random_state=seed)
    for base in learner.models:
        if hasattr(base,'n_jobs'):
            base.set_params(n_jobs=inner_jobs)
    specification = [{'class':type(base).__name__,
                      'parameters':{key:repr(value) for key,value in base.get_params().items()}}
                     for base in learner.models]
    learner.fit(xtr,labels)
    expected = [0,1,2,3] if role == 'joint' else [0,1]
    np.testing.assert_array_equal(learner.classes_,expected)
    result = {'probabilities':learner.predict_proba(xte),
              'specification':specification,'seconds':time.perf_counter()-started}
    if role == 'outcome':
        result['decisions'] = learner.predict(xte)
    return result


def run(dataset: str, input_path: Path, output: Path, seed: int, n_jobs: int) -> None:
    loader = importlib.import_module(f'analysis.analyze_{dataset}.run')
    xtr,xte,ytr,yte,gtr,gte = loader.load_data(input_path,seed)
    roles = min(3,n_jobs)
    inner_jobs = max(1,n_jobs//roles)
    tasks = [('outcome',np.asarray(ytr)),('group',np.asarray(gtr)),
             ('joint',_encode_joint(gtr,ytr))]
    LOG.info('Fitting shared nuisances: %s, train=%d, evaluation=%d, concurrent roles=%d, workers per learner=%d',
             dataset,len(xtr),len(xte),roles,inner_jobs)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        fits = Parallel(n_jobs=roles)(delayed(fit_predict)(role,xtr,xte,labels,seed,inner_jobs)
                                      for role,labels in tasks)
    outcome,group,joint = fits
    output.mkdir(parents=True,exist_ok=True)
    cache = output/'predictions.npz'
    np.savez_compressed(cache,probabilities=outcome['probabilities'][:,1],decisions=outcome['decisions'],
                        group_probabilities=group['probabilities'],joint_probabilities=joint['probabilities'],
                        outcomes=np.asarray(yte),groups=np.asarray(gte),
                        train_index=xtr.index.to_numpy(),test_index=xte.index.to_numpy())
    result = summarize_predictions(outcome['probabilities'][:,1],outcome['decisions'],
                                   group['probabilities'],joint['probabilities'],np.asarray(yte),np.asarray(gte))
    result['prediction_sha256'] = sha256(cache)
    with (output/'comparison.pkl').open('wb') as stream:
        pickle.dump(result,stream)
    write_json(output/'fit_metadata.json',{'dataset':dataset,'seed':seed,'raw_sha256':sha256(input_path),
               'preprocessing_sha256':sha256(Path(loader.__file__)), 'train_rows':len(xtr),'evaluation_rows':len(xte),
               'concurrent_roles':roles,'inner_jobs':inner_jobs,'prediction_sha256':sha256(cache),
               'roles':{role:{'learner_specification':fit['specification'],'fit_seconds':fit['seconds']}
                        for (role,_),fit in zip(tasks,fits)},
               'variance_convention':'Sample influence-function variance with ddof=1 for both targets.'})
    LOG.info('Saved matched comparison and prediction cache: %s',output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['adult','law'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--n-jobs',type=int,default=1)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error('--n-jobs must be positive')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    run(args.dataset,args.input,args.output_dir,args.seed,args.n_jobs)


if __name__ == '__main__':
    main()
