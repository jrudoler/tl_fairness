import pandas as pd
from joblib import Parallel, delayed
from experiments.nuisance_matched_rate import one_rep, HIGH, grid
from experiments.nuisance_training import one_rep as original, HIGH as OLD_HIGH, BOOST


def test_rate_change_reproduces_original_when_given_original_parameters():
    rows=one_rep('primary',100,0,n_eval=100,params=OLD_HIGH)
    previous=original('primary',100,100,0,cases=['higher_capacity'])
    pd.testing.assert_frame_equal(pd.DataFrame(rows),pd.DataFrame(previous))
    assert HIGH == dict(OLD_HIGH,learning_rate=BOOST['learning_rate'])
    assert len(grid())==16


def test_matched_rate_reuses_draws_and_is_worker_reproducible():
    small=dict(n_estimators=5,max_depth=5,learning_rate=.1)
    sequential=[one_rep('primary',100,r,n_eval=100,params=small) for r in range(2)]
    parallel=Parallel(n_jobs=2)(delayed(one_rep)('primary',100,r,n_eval=100,params=small) for r in range(2))
    for r,(a,b) in enumerate(zip(sequential,parallel)):
        pd.testing.assert_frame_equal(pd.DataFrame(a),pd.DataFrame(b))
        baseline=original('primary',100,100,r,cases=['default'])
        assert a[0]['data_hash']==baseline[0]['data_hash']
        assert a[0]['outcome_hash']==a[1]['outcome_hash']
