import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from experiments.nuisance_early_stopping import one_rep, BOOST, HIGH, EARLY_STOPPING
from experiments.nuisance_training import one_rep as baseline, draw


def test_early_stopping_reuses_draws_and_shared_fits_across_workers():
    cases=['default','higher_capacity','boost_linear','linear_boost','linear_linear']
    args=[('primary',100,100,r) for r in range(2)]
    seq=[one_rep(*a,cases=cases) for a in args]
    par=Parallel(n_jobs=2)(delayed(one_rep)(*a,cases=cases) for a in args)
    for a,b in zip(seq,par): pd.testing.assert_frame_equal(pd.DataFrame(a),pd.DataFrame(b))
    for r,rows in enumerate(seq):
        df=pd.DataFrame(rows)
        old=pd.DataFrame(baseline(*args[r],cases=cases))
        assert df.data_hash.eq(old.data_hash).all()
        for role in ['outcome','group']:
            assert df.groupby(role+'_model')[role+'_hash'].nunique().eq(1).all()
            linear=df[role+'_model']=='linear'
            assert df.loc[linear,role+'_hash'].eq(old.loc[linear,role+'_hash']).all()
            for name,cap in [('default',100),('higher_capacity',1000)]:
                counts=df.loc[df[role+'_model']==name,role+'_trees']
                assert counts.between(1,cap).all()
        assert np.isfinite(df[['outcome_probability_mse','group_probability_mse']]).all().all()


def test_builtin_stopping_does_not_use_evaluation_observations():
    x,g,y,_,_=draw(300,np.random.default_rng(123))
    a=GradientBoostingClassifier(**HIGH,random_state=17).fit(x[:100],y[:100])
    n=a.n_estimators_
    a.predict_proba(x[100:]);a.predict_proba(x[100:]+100)
    b=GradientBoostingClassifier(**HIGH,random_state=17).fit(x[:100],y[:100])
    assert n==b.n_estimators_ < 1000
    np.testing.assert_array_equal(a.predict_proba(x[100:]),b.predict_proba(x[100:]))
    assert BOOST['learning_rate']==HIGH['learning_rate']==.1
    for k,v in EARLY_STOPPING.items(): assert BOOST[k]==HIGH[k]==v
