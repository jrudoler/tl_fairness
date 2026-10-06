"""Replace only higher-capacity fits with learning rate 0.1 on the original draws.

The original experiment stays frozen. Checkpoints and collected outputs are separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.nuisance_training import (
    SCENARIOS, SIZES, CONTROL_SIZES, METHODS, KEYS, BOOST, draw,
    _prob_group_contrast, model_interval, summarize, source_hashes,
)
HIGH = dict(n_estimators=1000, max_depth=5, learning_rate=0.1)


def grid():
    return [(s,n) for n in reversed(SIZES) for s in SCENARIOS
            if s == 'primary' or n in CONTROL_SIZES]


def one_rep(scenario, n_train, replicate, n_eval=2000, seed=20261005, params=None):
    params = HIGH if params is None else params
    sid = list(SCENARIOS).index(scenario)
    rng = np.random.default_rng(np.random.SeedSequence([seed,sid,n_train,n_eval,replicate]))
    x,g,y,_,_ = draw(n_train+n_eval,rng,scenario)
    predictions = {}
    for role,response,j in [('outcome',y,0),('group',g,1)]:
        ms = int(np.random.SeedSequence([seed,sid,n_train,n_eval,replicate,j]).generate_state(1)[0])
        fit = GradientBoostingClassifier(**params,random_state=ms).fit(x[:n_train],response[:n_train])
        predictions[role] = fit.predict_proba(x[n_train:])[:,1]
    dh,ph = predictions['outcome'],predictions['group']
    ge,ye = g[n_train:],y[n_train:]
    estimates = [_prob_group_contrast(dh,ye-dh,np.column_stack([1-ph,ph]),np.column_stack([ge==0,ge==1])),model_interval(dh,ge)]
    dhash = hashlib.sha256(x.astype('<f8').tobytes()+g.tobytes()+y.tobytes()).hexdigest()
    hashes = {k:hashlib.sha256(v.astype('<f8').tobytes()).hexdigest() for k,v in predictions.items()}
    return [dict(scenario=scenario,train_size=n_train,test_size=n_eval,replicate=replicate,
                 case='higher_capacity',outcome_model='higher_capacity',group_model='higher_capacity',
                 method=method,estimate=estimate,ci_low=lo,ci_high=hi,se=(hi-lo)/3.92,width=hi-lo,
                 data_hash=dhash,outcome_hash=hashes['outcome'],group_hash=hashes['group'])
            for method,(estimate,(lo,hi)) in zip(METHODS,estimates)]


def run_shard(output, scenario, n_train, rep_start=0, reps=25, n_jobs=1):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    hashes = source_hashes()
    hashes['experiments/nuisance_matched_rate.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta = dict(scenario=scenario,train_size=n_train,test_size=2000,rep_start=rep_start,reps=reps,
                seed=20261005,higher_capacity_learner=HIGH,default_learner=BOOST,
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                sklearn=sklearn.__version__,source_hashes=hashes)
    path = output/'metadata.json'
    if path.exists() and json.loads(path.read_text()) != meta:
        raise ValueError('Refusing incompatible matched-rate resume')
    path.write_text(json.dumps(meta,indent=2)+'\n')
    pending = [r for r in range(rep_start,rep_start+reps) if not (output/f'rep_{r:05d}.csv').exists()]
    for rows in Parallel(n_jobs=n_jobs,return_as='generator_unordered',batch_size=1)(
            delayed(one_rep)(scenario,n_train,r) for r in pending):
        r = rows[0]['replicate']; temp = output/f'rep_{r:05d}.csv.tmp'
        pd.DataFrame(rows).to_csv(temp,index=False); temp.replace(output/f'rep_{r:05d}.csv')
        print(f'{scenario} n={n_train} replicate={r} complete',flush=True)


def collect(shards,base,output):
    shards,base,output = map(Path,[shards,base,output])
    replacement = pd.concat([pd.read_csv(p) for p in sorted(shards.rglob('rep_*.csv'))],ignore_index=True)
    expected = {(s,n,r,m) for s,n in grid() for r in range(1000) for m in METHODS}
    assert len(replacement)==32000
    assert set(replacement[['scenario','train_size','replicate','method']].itertuples(index=False,name=None))==expected
    original = pd.read_csv(base/'replicates.csv.gz')
    old = original[original.case=='higher_capacity']
    ids = ['scenario','train_size','replicate','method']
    matched = replacement.merge(old[ids+['data_hash']],on=ids,suffixes=('','_original'),validate='one_to_one')
    assert matched.data_hash.eq(matched.data_hash_original).all(), 'Original draws changed'
    population = pd.read_csv(base/'population.csv')
    raw,summary,paired = summarize(pd.concat([original[original.case!='higher_capacity'],replacement],ignore_index=True),population)
    for scenario,part in summary.groupby('scenario'):
        assert population.set_index('scenario').loc[scenario,'integration_se'] < .01*part.bias_mcse.min()
    output.mkdir(parents=True,exist_ok=True)
    raw.to_csv(output/'replicates.csv.gz',index=False)
    summary.to_csv(output/'summary.csv',index=False)
    paired.to_csv(output/'paired_comparisons.csv',index=False)
    population.to_csv(output/'population.csv',index=False)
    manifest = dict(base_manifest=json.loads((base/'manifest.json').read_text()),
                    replacement_case='higher_capacity',higher_capacity_learner=HIGH,
                    matched_rate_shards={str(p.relative_to(shards)):json.loads(p.read_text()) for p in sorted(shards.rglob('metadata.json'))})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    differences=[]
    for (s,n,m),part in raw[raw.case=='higher_capacity'].groupby(['scenario','train_size','method']):
        a=part.set_index('replicate')
        b=old[(old.scenario==s)&(old.train_size==n)&(old.method==m)].set_index('replicate').loc[a.index]
        row=dict(scenario=s,train_size=n,method=m,comparison='rate 0.1 minus rate 0.05',reps=len(a))
        for name,col in [('coverage','covered'),('bias','error')]:
            diff=a[col].astype(float)-b[col].astype(float)
            row[name+'_difference']=diff.mean();row[name+'_difference_mcse']=diff.sem()
        differences.append(row)
    pd.DataFrame(differences).to_csv(output/'rate_comparisons.csv',index=False)
    print(f'Collected {len(raw):,} rows; replaced 16,000 higher-capacity fits.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--array-index',type=int,required=True)
    run.add_argument('--n-jobs',type=int,default=1)
    run.add_argument('--output-dir',default='data/generated/nuisance_matched_rate_shards')
    merge=sub.add_parser('collect');merge.add_argument('--input-dir',default='data/generated/nuisance_matched_rate_shards')
    merge.add_argument('--base-dir',default='data/generated/nuisance_training')
    merge.add_argument('--output-dir',default='data/generated/nuisance_matched_rate')
    a=p.parse_args()
    if a.command=='collect': collect(a.input_dir,a.base_dir,a.output_dir)
    else:
        if not 0 <= a.array_index < 640: p.error('Array index must be 0..639')
        config,shard=divmod(a.array_index,40);s,n=grid()[config]
        run_shard(Path(a.output_dir)/f'{s}_{n}_{shard:02d}',s,n,rep_start=25*shard,n_jobs=a.n_jobs)
