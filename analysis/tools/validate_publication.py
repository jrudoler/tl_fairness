"""Validate frozen publication inputs without scheduling or modifying simulations."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from tlfair.provenance import sha256, runtime, write_json
from analysis.tools.validate import validate_csv, validate_results_pickle, validate_truth_pickle, EXPECTED_CSVS
from analysis.nuisance_tables.validate import validate as validate_nuisance


def verify_inputs(registry, root=ROOT):
    for item in registry['inputs']:
        path=root/item['path']
        if not path.is_file():
            raise FileNotFoundError(f'Missing publication input {path}; restore the recorded bundle, not a new simulation.')
        if sha256(path)!=item['sha256']:
            raise ValueError(f'Publication input checksum mismatch: {path}')


def validate(deep=False):
    registry=json.loads((ROOT/'config/publication.json').read_text())
    verify_inputs(registry)
    directory=ROOT/'data/generated'
    for name,columns in EXPECTED_CSVS.items():
        if name.startswith('cmi_'):validate_csv(directory/name,columns)
    validate_truth_pickle(directory/'cmi_population_truth.pkl')
    for dataset in ['adult','law']:
        validate_results_pickle(directory/f'{dataset}_results.pkl',
                                ['parity','prob_parity','opportunity','prob_opp','cmi'])
    cond=pd.read_csv(directory/'cmi_conditioning_set.csv')
    assert len(cond)==12 and cond.reps.eq(200).all()
    assert set(cond.sample_size)=={2500,5000}
    assert not cond.duplicated(['sample_size','n_confounders_included']).any()
    assert cond.groupby('sample_size').n_confounders_included.apply(lambda x:set(x)==set(range(6))).all()
    assert np.isfinite(cond[['mean_estimate','mean_ci_width','reject_rate']]).all().all()
    assert cond.reject_rate.between(0,1).all() and cond.mean_ci_width.ge(0).all()
    null=cond[cond.n_confounders_included==5]
    assert null.coverage_at_null.between(0,1).all()
    summary=pd.read_csv(directory/'parity_retraining/summary.csv')
    keys=['effect','train_size','test_size','method']
    assert len(summary)==405 and not summary.duplicated(keys).any()
    assert summary.training_reps.eq(1000).all() and summary.evaluation_reps.eq(50).all()
    assert summary.coverage_data.between(0,1).all() and summary.rejection_rate.between(0,1).all()
    validate_nuisance(directory/'nuisance_early_stopping')
    if deep:
        from experiments.parity_retraining import summarize
        rebuilt=[]
        for i,part in enumerate(pd.read_csv(directory/'parity_retraining/replicates.csv.gz',chunksize=150000)):
            assert len(part)==150000
            assert len(part[['effect','train_size','test_size']].drop_duplicates())==1
            assert part.groupby(['method','replicate']).size().eq(50).all()
            rebuilt.append(summarize(part))
            if i%15==0:print(f'Validated retraining block {i+1}/135',flush=True)
        actual=pd.concat(rebuilt,ignore_index=True).sort_values(keys).reset_index(drop=True)
        expected=summary.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-10,rtol=1e-8)
    return dict(publication_id=registry['publication_id'],input_count=len(registry['inputs']),
                registry_sha256=sha256(ROOT/'config/publication.json'),runtime=runtime(),
                retraining_summary_rows=len(summary),retraining_raw_verified=deep,
                nuisance_rows=118000,conditioning_rows=len(cond),status='passed',
                limitations=registry['limitations'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results/provenance/validation.json')
    args=parser.parse_args()
    result=validate(args.deep);write_json(args.output,result);print(json.dumps(result,indent=2))
