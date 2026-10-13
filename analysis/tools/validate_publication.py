"""Validate frozen publication inputs without scheduling or modifying simulations."""
import argparse
import json
import pickle
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from tlfair.provenance import sha256, runtime, write_json
from analysis.tools.validate import validate_csv, validate_results_pickle, validate_truth_pickle, EXPECTED_CSVS
from analysis.nuisance_tables.validate import validate as validate_nuisance
from analysis.tools.audit_publication_runs import audit


def validate_cmi_records() -> dict[str, int]:
    """Reconstruct manuscript CMI summaries from independent saved replicates."""
    selections = json.loads((ROOT/'config/publication-runs.json').read_text())['analyses']
    specifications = json.loads((ROOT/'config/analyses.json').read_text())['analyses']
    estimation = ROOT/selections['cmi_estimation']['run']
    parameters = specifications['cmi_estimation']['parameters']
    with (ROOT/'data/generated/cmi_population_truth.pkl').open('rb') as stream:
        truth = pickle.load(stream)
    records = pd.read_csv(estimation/'coverage_replicates.csv.gz')
    keys = ['sample_size','kappa']
    assert not records.duplicated(keys+['replicate']).any()
    assert records.groupby(keys).replicate.apply(lambda r:set(r)==set(range(parameters['sims']))).all()
    covered = (records.ci_low <= records.kappa.map(truth)) & (records.ci_high >= records.kappa.map(truth))
    np.testing.assert_array_equal(records.covered,covered.astype(float))
    rebuilt = records.groupby(keys).agg(coverage=('covered','mean'), estimate=('estimate','mean')).reset_index()
    rebuilt['error'] = rebuilt.estimate - rebuilt.kappa.map(truth)
    expected = pd.read_csv(ROOT/'data/generated/cmi_estimation_coverage.csv').sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,rebuilt[expected.columns],check_dtype=False,atol=1e-12,rtol=1e-10)
    comparisons = pd.read_csv(estimation/'comparison_replicates.csv.gz')
    assert not comparisons.duplicated(keys+['replicate']).any()
    assert comparisons.groupby(keys).replicate.apply(lambda r:set(r)==set(range(parameters['compare-repeats']))).all()
    rows = []
    for (size,kappa),part in comparisons.groupby(keys):
        for column,label in [('tl','TL'),('knn','KNN')]:
            rows.append({'sample size':size,'kappa':kappa,'type':label,
                         'mean':part[column].mean(),'bottom_five':part[column].quantile(.05),
                         'top_five':part[column].quantile(.95)})
    compare_keys = ['sample size','type','kappa']
    expected = pd.read_csv(ROOT/'data/generated/cmi_estimation_comparison.csv').sort_values(compare_keys).reset_index(drop=True)
    rebuilt = pd.DataFrame(rows).sort_values(compare_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,rebuilt[expected.columns],check_dtype=False,atol=1e-12,rtol=1e-10)
    conditioning = ROOT/selections['cmi_conditioning_set']['run']
    params = specifications['cmi_conditioning_set']['parameters']
    conditional_records = pd.read_csv(conditioning/'replicates.csv.gz')
    keys = ['sample_size','n_confounders_included']
    assert not conditional_records.duplicated(keys+['replicate']).any()
    assert conditional_records.groupby(keys).replicate.apply(lambda r:set(r)==set(range(params['reps']))).all()
    np.testing.assert_allclose(conditional_records.ci_low,conditional_records.estimate-1.96*conditional_records.se)
    np.testing.assert_allclose(conditional_records.ci_high,conditional_records.estimate+1.96*conditional_records.se)
    np.testing.assert_allclose(conditional_records.lower_one_sided,conditional_records.estimate-1.645*conditional_records.se)
    np.testing.assert_array_equal(conditional_records.rejects_zero,(conditional_records.lower_one_sided>0).astype(float))
    conditional_records['width'] = conditional_records.ci_high - conditional_records.ci_low
    conditional_records['covered'] = ((conditional_records.ci_low<=0)&(conditional_records.ci_high>=0)).astype(float)
    rebuilt = conditional_records.groupby(keys).agg(reject_rate=('rejects_zero','mean'),
                  mean_estimate=('estimate','mean'),mean_ci_width=('width','mean'),coverage_at_null=('covered','mean')).reset_index()
    rebuilt.loc[rebuilt.n_confounders_included != params['n-confounders'],'coverage_at_null'] = np.nan
    expected = pd.read_csv(ROOT/'data/generated/cmi_conditioning_set.csv').sort_values(keys).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected[rebuilt.columns],rebuilt,check_dtype=False,atol=1e-12,rtol=1e-10)
    return {'coverage_replicates':len(records),'comparison_replicates':len(comparisons),
            'conditioning_replicates':len(conditional_records)}


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
    lineage = audit()
    if lineage['status'] != 'verified':
        raise ValueError('Publication computation lineage is not verified: '+json.dumps(lineage))
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
    cmi_records = validate_cmi_records()
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
                cmi_records=cmi_records,computation_lineage='verified',
                limitations=registry['limitations'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results/provenance/validation.json')
    args=parser.parse_args()
    result=validate(args.deep);write_json(args.output,result);print(json.dumps(result,indent=2))
