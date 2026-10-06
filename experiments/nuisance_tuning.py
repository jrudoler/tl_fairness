"""Training-only CV extension of the frozen matched nuisance experiment.

Three stratified folds select predictive log loss separately for D and pi.
The fixed search includes both published configurations and shallow alternatives.
Staged predictions share fits across tree counts; no evaluation labels tune models.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.nuisance_training import (draw, METHODS, SIZES, BOOST, HIGH,
    _prob_group_contrast, model_interval, summarize, source_hashes)

GRID = [dict(n_estimators=n, max_depth=d, learning_rate=.05)
        for d in [1, 3, 5] for n in [100, 500, 1000]] + [BOOST]
CV_FOLDS = 3


def fit_tuned(x, response, evaluate, model_seed, cv_seed, grid=GRID):
    """Return predictions and a complete CV audit, using training observations only."""
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=cv_seed).split(x, response))
    scores = {json.dumps(p, sort_keys=True): [] for p in grid}
    paths = sorted({(p['max_depth'], p['learning_rate']) for p in grid})
    for depth, rate in paths:
        counts = sorted({p['n_estimators'] for p in grid if p['max_depth']==depth and p['learning_rate']==rate})
        for train, valid in folds:
            fit = GradientBoostingClassifier(n_estimators=max(counts), max_depth=depth,
                    learning_rate=rate, random_state=model_seed).fit(x[train], response[train])
            for n, proba in enumerate(fit.staged_predict_proba(x[valid]), 1):
                if n not in counts:
                    continue
                prob = np.clip(proba[:,1], np.finfo(float).eps, 1-np.finfo(float).eps)
                yy = response[valid]
                loss = -(yy*np.log(prob)+(1-yy)*np.log1p(-prob)).mean()
                key = json.dumps(dict(n_estimators=n,max_depth=depth,learning_rate=rate),sort_keys=True)
                scores[key].append((float(loss),len(valid)))
    records = []
    for key, losses in scores.items():
        records.append(dict(params=json.loads(key), fold_log_losses=[s for s,_ in losses],
                            mean_log_loss=float(np.average([s for s,_ in losses],weights=[n for _,n in losses]))))
    # Stable tie rule favors fewer terminal nodes, then fewer trees and lower rate.
    selected = min(records, key=lambda r:(r['mean_log_loss'],
        r['params']['n_estimators']*2**r['params']['max_depth'],
        r['params']['n_estimators'],r['params']['learning_rate']))
    fit = GradientBoostingClassifier(**selected['params'],random_state=model_seed).fit(x,response)
    predictions = fit.predict_proba(evaluate)[:,1]
    return predictions, dict(selected=selected['params'], selected_log_loss=selected['mean_log_loss'],
                            candidates=records, cv_seed=cv_seed, model_seed=model_seed)


def one_rep(n_train, replicate, n_eval=2000, seed=20261005, grid=GRID):
    started=time.monotonic()
    rng=np.random.default_rng(np.random.SeedSequence([seed,0,n_train,n_eval,replicate]))
    x,g,y,_,_=draw(n_train+n_eval,rng,'primary')
    predictions={};audit={}
    for role,response,j in [('outcome',y,0),('group',g,1)]:
        ms=int(np.random.SeedSequence([seed,0,n_train,n_eval,replicate,j]).generate_state(1)[0])
        cs=int(np.random.SeedSequence([seed,0,n_train,n_eval,replicate,j,731]).generate_state(1)[0])
        predictions[role],audit[role]=fit_tuned(x[:n_train],response[:n_train],x[n_train:],ms,cs,grid)
    dh,ph=predictions['outcome'],predictions['group'];ge,ye=g[n_train:],y[n_train:]
    estimates=[_prob_group_contrast(dh,ye-dh,np.column_stack([1-ph,ph]),np.column_stack([ge==0,ge==1])),model_interval(dh,ge)]
    data_hash=hashlib.sha256(x.astype('<f8').tobytes()+g.tobytes()+y.tobytes()).hexdigest()
    hashes={k:hashlib.sha256(v.astype('<f8').tobytes()).hexdigest() for k,v in predictions.items()}
    rows=[]
    for method,(estimate,(lo,hi)) in zip(METHODS,estimates):
        rows.append(dict(scenario='primary',train_size=n_train,test_size=n_eval,replicate=replicate,
            case='cross_validated',outcome_model='cross_validated',group_model='cross_validated',
            method=method,estimate=estimate,ci_low=lo,ci_high=hi,se=(hi-lo)/3.92,width=hi-lo,
            data_hash=data_hash,outcome_hash=hashes['outcome'],group_hash=hashes['group']))
    audit.update(replicate=replicate,train_size=n_train,test_size=n_eval,elapsed_seconds=time.monotonic()-started)
    return rows,audit


def run_shard(output,n_train,rep_start=0,reps=25,n_jobs=1):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();hashes['experiments/nuisance_tuning.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta=dict(scenario='primary',train_size=n_train,test_size=2000,rep_start=rep_start,reps=reps,
        seed=20261005,grid=GRID,cv_folds=CV_FOLDS,criterion='stratified out-of-fold log loss',
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,source_hashes=hashes)
    path=output/'metadata.json'
    if path.exists() and json.loads(path.read_text())!=meta:
        raise ValueError('Refusing incompatible tuning resume')
    path.write_text(json.dumps(meta,indent=2)+'\n')
    pending=[r for r in range(rep_start,rep_start+reps) if not (output/f'rep_{r:05d}.csv').exists()]
    for rows,audit in Parallel(n_jobs=n_jobs,return_as='generator_unordered',batch_size=1)(delayed(one_rep)(n_train,r) for r in pending):
        r=rows[0]['replicate']
        (output/f'tuning_{r:05d}.json').write_text(json.dumps(audit,indent=2)+'\n')
        tmp=output/f'rep_{r:05d}.csv.tmp';pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(output/f'rep_{r:05d}.csv')
        print(f'n={n_train} replicate={r} complete ({audit["elapsed_seconds"]:.1f}s)',flush=True)


def collect(shards,base,output):
    shards,base,output=map(Path,[shards,base,output])
    tuned=pd.concat([pd.read_csv(p) for p in sorted(shards.rglob('rep_*.csv'))],ignore_index=True)
    expected={(n,r,m) for n in SIZES for r in range(1000) for m in METHODS}
    assert len(tuned)==14000 and set(tuned[['train_size','replicate','method']].itertuples(index=False,name=None))==expected
    raw=pd.read_csv(base/'replicates.csv.gz')
    matching=raw[(raw.scenario=='primary')&(raw.case=='default')][['train_size','replicate','method','data_hash']]
    joined=tuned.merge(matching,on=['train_size','replicate','method'],suffixes=('','_original'),validate='one_to_one')
    assert joined.data_hash.eq(joined.data_hash_original).all(), 'Evaluation/training draws differ from original comparisons'
    assert tuned.groupby(['train_size','replicate']).outcome_hash.nunique().eq(1).all()
    population=pd.read_csv(base/'population.csv')
    raw,summary,paired=summarize(pd.concat([raw,tuned],ignore_index=True),population)
    assert population.set_index('scenario').loc['primary','integration_se'] < .01*summary[summary.scenario=='primary'].bias_mcse.min()
    output.mkdir(parents=True,exist_ok=True)
    raw.to_csv(output/'replicates.csv.gz',index=False);summary.to_csv(output/'summary.csv',index=False)
    paired.to_csv(output/'paired_comparisons.csv',index=False);population.to_csv(output/'population.csv',index=False)
    audits=[json.loads(p.read_text()) for p in sorted(shards.rglob('tuning_*.json'))]
    assert len(audits)==7000
    selections=[]
    for audit in audits:
        for role in ['outcome','group']:
            record=audit[role]
            selections.append(dict(train_size=audit['train_size'],replicate=audit['replicate'],role=role,
                **record['selected'],cv_log_loss=record['selected_log_loss']))
    pd.DataFrame(selections).to_csv(output/'tuning_selections.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(dict(base_manifest=json.loads((base/'manifest.json').read_text()),
        tuning_shards={str(p.relative_to(shards)):json.loads(p.read_text()) for p in sorted(shards.rglob('metadata.json'))}),indent=2)+'\n')
    # All between-learner numerical comparisons preserve the paired dataset draws.
    differences=[]
    primary=raw[raw.scenario=='primary']
    for (n,method),part in primary.groupby(['train_size','method']):
        a=part[part.case=='cross_validated'].set_index('replicate')
        for other in ['default','higher_capacity']:
            b=part[part.case==other].set_index('replicate').loc[a.index]
            row=dict(train_size=n,method=method,comparison='cross_validated minus '+other,reps=len(a))
            for name,col in [('coverage','covered'),('bias','error')]:
                diff=a[col].astype(float)-b[col].astype(float)
                row[name+'_difference']=diff.mean();row[name+'_difference_mcse']=diff.sem()
            differences.append(row)
    pd.DataFrame(differences).to_csv(output/'learner_comparisons.csv',index=False)
    print(f'Collected {len(raw):,} records, with 7,000 matched tuned fits.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--output-dir',required=True);run.add_argument('--train-size',type=int,required=True)
    run.add_argument('--rep-start',type=int,default=0);run.add_argument('--reps',type=int,default=25);run.add_argument('--n-jobs',type=int,default=1)
    merge=sub.add_parser('collect');merge.add_argument('--input-dir',required=True);merge.add_argument('--base-dir',default='data/generated/nuisance_training');merge.add_argument('--output-dir',default='data/generated/nuisance_training_cv')
    a=p.parse_args()
    if a.command=='run': run_shard(a.output_dir,a.train_size,a.rep_start,a.reps,a.n_jobs)
    else: collect(a.input_dir,a.base_dir,a.output_dir)
