import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from experiments.nuisance_tuning import fit_tuned, one_rep, GRID
from experiments.nuisance_training import one_rep as fixed_rep, BOOST, HIGH

SMALL=[dict(n_estimators=n,max_depth=d,learning_rate=.1) for n in [5,10] for d in [1,2]]

def test_cv_staged_scores_equal_separate_fits_and_ignore_evaluation():
    rng=np.random.default_rng(10);x=rng.normal(size=(90,5));y=np.tile([0,1],45)
    _,audit=fit_tuned(x,y,x[:10],11,12,SMALL)
    _,changed=fit_tuned(x,y,100+x[:7],11,12,SMALL)
    assert audit==changed
    folds=list(StratifiedKFold(3,shuffle=True,random_state=12).split(x,y))
    for candidate in audit['candidates']:
        expected=[]
        for train,valid in folds:
            fit=GradientBoostingClassifier(**candidate['params'],random_state=11).fit(x[train],y[train])
            expected.append(log_loss(y[valid],fit.predict_proba(x[valid])))
        np.testing.assert_allclose(candidate['fold_log_losses'],expected)
    assert audit['selected_log_loss']==min(c['mean_log_loss'] for c in audit['candidates'])
    assert BOOST in GRID and HIGH in GRID

def test_tuned_reuses_original_draws_and_is_worker_reproducible():
    sequential=[one_rep(100,r,n_eval=100,grid=SMALL) for r in range(2)]
    parallel=Parallel(n_jobs=2)(delayed(one_rep)(100,r,n_eval=100,grid=SMALL) for r in range(2))
    for r,((rows,audit),(other,other_audit)) in enumerate(zip(sequential,parallel)):
        pd.testing.assert_frame_equal(pd.DataFrame(rows),pd.DataFrame(other))
        assert audit['outcome']==other_audit['outcome'] and audit['group']==other_audit['group']
        original=fixed_rep('primary',100,100,r,cases=['default'])
        assert {row['data_hash'] for row in rows}=={row['data_hash'] for row in original}
        assert rows[0]['outcome_hash']==rows[1]['outcome_hash']
