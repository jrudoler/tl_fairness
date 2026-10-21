"""Validate frozen publication inputs without scheduling or modifying simulations."""
import argparse
import importlib
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


def validate_matched_predictions() -> dict[str, int]:
    """Reconstruct both targets from the exact shared held-out prediction cache."""
    from analysis.matched_real_data.run import summarize_predictions
    selections = json.loads((ROOT/'config/publication-runs.json').read_text())['analyses']
    sizes = {}
    for dataset in ['adult', 'law']:
        if f'model_fairness_{dataset}' not in selections:
            continue
        directory = ROOT/selections[f'model_fairness_{dataset}']['run']
        metadata = json.loads((directory/'fit_metadata.json').read_text())
        with (ROOT/f'data/generated/{dataset}_model_comparison.pkl').open('rb') as stream:
            expected = pickle.load(stream)
        cache = directory/'predictions.npz'
        assert sha256(cache) == metadata['prediction_sha256'] == expected['prediction_sha256']
        loader = importlib.import_module(f'analysis.analyze_{dataset}.run')
        raw = ROOT/f'data/raw/{dataset}.csv'
        assert metadata['raw_sha256'] == sha256(raw)
        assert metadata['preprocessing_sha256'] == sha256(Path(loader.__file__))
        xtr, xte, _, yte, _, gte = loader.load_data(raw, metadata['seed'])
        assert metadata['dataset'] == dataset
        assert metadata['train_rows'] == len(xtr)
        assert metadata['evaluation_rows'] == len(xte)
        with np.load(cache) as saved:
            for key, value in [('train_index', xtr.index), ('test_index', xte.index),
                               ('outcomes', yte), ('groups', gte)]:
                np.testing.assert_array_equal(saved[key], value)
            for key in ['probabilities', 'group_probabilities', 'joint_probabilities']:
                assert np.isfinite(saved[key]).all()
                assert ((saved[key] >= 0) & (saved[key] <= 1)).all()
            for key in ['group_probabilities', 'joint_probabilities']:
                np.testing.assert_allclose(saved[key].sum(axis=1), 1)
            np.testing.assert_array_equal(saved['decisions'], (saved['probabilities'] > .5).astype(int))
            rebuilt = summarize_predictions(**{key:saved[key] for key in [
                'probabilities', 'decisions', 'group_probabilities', 'joint_probabilities', 'outcomes', 'groups']})
        for target in ['data_fairness', 'model_fairness']:
            assert set(expected[target]) == set(rebuilt[target])
            for metric, (estimate, interval) in rebuilt[target].items():
                actual = expected[target][metric]
                np.testing.assert_allclose([actual[0], *actual[1]], [estimate, *interval], atol=1e-12, rtol=1e-10)
        for metric in ['parity', 'opportunity']:
            assert expected['data_fairness'][metric] == expected['model_fairness'][metric]
        sizes[dataset] = len(xte)
    return sizes


def validate_cmi_fit_protocol(directory: Path, records: pd.DataFrame) -> None:
    """Certify the shared depth-3 protocol and actual fitted model sizes."""
    from tlfair.cmi_estimation import CMI_LEARNER_PARAMETERS
    record = json.loads((directory/'run_manifest.json').read_text())
    learner = record['design']['cmi_learner']
    assert learner['class'] == 'GradientBoostingClassifier'
    assert learner['calibration'] is None and learner['full_training_refit'] is False
    for key, value in CMI_LEARNER_PARAMETERS.items():
        assert learner['parameters'][key] == value, (key, learner['parameters'][key], value)
    fitted = records[records.estimator != 'knn']
    assert fitted.boosting_rounds.between(1, 100).all()
    assert (fitted.boosting_rounds % 1 == 0).all()
    np.testing.assert_array_equal(fitted.tree_count, 4*fitted.boosting_rounds)
    assert fitted.model_seed.between(0, 2**31-1).all()


def validate_cmi_runs(runs: dict[str, Path], canonical: Path | None = None) -> dict[str, int]:
    """Reconstruct paired CMI summaries and reject any unsuccessful targeting."""
    from tlfair.cmi_estimation import require_convergence
    specifications = json.loads((ROOT/'config/analyses.json').read_text())['analyses']
    estimation = runs['cmi_estimation']
    params = specifications['cmi_estimation']['parameters']
    with (estimation/'population_truth.pkl').open('rb') as stream:
        truth = pickle.load(stream)
    assert set(truth) == set(params['weights']) and truth[0] == 0
    records = pd.read_csv(estimation/'coverage_replicates.csv.gz')
    keys = ['sample_size','kappa','estimator']
    assert set(records.estimator) == {'one_step','substitution','tmle'}
    assert not records.duplicated(keys+['replicate']).any()
    assert len(records) == len(params['sizes'])*len(params['weights'])*params['sims']*3
    assert set(records.sample_size) == set(params['sizes']) and set(records.kappa) == set(params['weights'])
    assert records.groupby(keys).replicate.apply(lambda r:set(r)==set(range(params['sims']))).all()
    require_convergence(records)
    validate_cmi_fit_protocol(estimation, records)
    np.testing.assert_allclose(records.ci_low, records.estimate-1.96*records.se)
    np.testing.assert_allclose(records.ci_high, records.estimate+1.96*records.se)
    records['covered'] = ((records.ci_low<=records.kappa.map(truth)) &
                          (records.ci_high>=records.kappa.map(truth))).astype(float)
    rebuilt = records.groupby(keys).agg(coverage=('covered','mean'),estimate=('estimate','mean')).reset_index()
    rebuilt['error'] = rebuilt.estimate-rebuilt.kappa.map(truth)
    expected = pd.read_csv(canonical/'cmi_estimation_coverage.csv' if canonical else estimation/'coverage.csv')
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
        rebuilt[expected.columns].sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-10)
    comparisons = pd.read_csv(estimation/'comparison_replicates.csv.gz')
    assert set(comparisons.estimator) == {'one_step','substitution','tmle','knn'}
    assert not comparisons.duplicated(keys+['replicate']).any()
    assert len(comparisons) == len(params['sizes'])*len(params['weights'])*params['compare-repeats']*4
    assert comparisons.groupby(keys).replicate.apply(lambda r:set(r)==set(range(params['compare-repeats']))).all()
    require_convergence(comparisons)
    validate_cmi_fit_protocol(estimation, comparisons)
    rows = []
    labels = {'tmle':'TMLE','one_step':'One-step','substitution':'Substitution','knn':'KNN'}
    for (size,kappa,method),part in comparisons.groupby(keys):
        rows.append({'sample size':size,'kappa':kappa,'estimator':method,'type':labels[method],
                     'mean':part.estimate.mean(),'bottom_five':part.estimate.quantile(.05),
                     'top_five':part.estimate.quantile(.95)})
    compare_keys = ['sample size','type','kappa']
    expected = pd.read_csv(canonical/'cmi_estimation_comparison.csv' if canonical else estimation/'comparison.csv')
    rebuilt = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(expected.sort_values(compare_keys).reset_index(drop=True),
        rebuilt[expected.columns].sort_values(compare_keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-10)
    conditioning = runs['cmi_conditioning_set']
    params = specifications['cmi_conditioning_set']['parameters']
    conditional = pd.read_csv(conditioning/'replicates.csv.gz')
    keys = ['sample_size','n_confounders_included','estimator']
    assert not conditional.duplicated(keys+['replicate']).any()
    assert set(conditional.estimator) == {'one_step','substitution','tmle'}
    assert len(conditional) == len(params['sizes'])*(params['n-confounders']+1)*params['reps']*3
    assert conditional.groupby(keys).replicate.apply(lambda r:set(r)==set(range(params['reps']))).all()
    require_convergence(conditional)
    validate_cmi_fit_protocol(conditioning, conditional)
    np.testing.assert_allclose(conditional.ci_low,conditional.estimate-1.96*conditional.se)
    np.testing.assert_allclose(conditional.ci_high,conditional.estimate+1.96*conditional.se)
    np.testing.assert_allclose(conditional.lower_one_sided,conditional.estimate-1.645*conditional.se)
    np.testing.assert_array_equal(conditional.rejects_zero,(conditional.lower_one_sided>0).astype(float))
    conditional['width'] = conditional.ci_high-conditional.ci_low
    conditional['covered'] = ((conditional.ci_low<=0)&(conditional.ci_high>=0)).astype(float)
    rebuilt = conditional.groupby(keys).agg(reject_rate=('rejects_zero','mean'),
        mean_estimate=('estimate','mean'),mean_ci_width=('width','mean'),coverage_at_null=('covered','mean')).reset_index()
    rebuilt.loc[rebuilt.n_confounders_included != params['n-confounders'],'coverage_at_null'] = np.nan
    expected = pd.read_csv(canonical/'cmi_conditioning_set.csv' if canonical else conditioning/'summary.csv')
    pd.testing.assert_frame_equal(expected[rebuilt.columns].sort_values(keys).reset_index(drop=True),
        rebuilt.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-10)
    return {'coverage_replicates':len(records),'comparison_replicates':len(comparisons),
            'conditioning_replicates':len(conditional)}


def validate_cmi_real_data(runs: dict[str, Path], canonical: Path | None = None) -> dict[str, int]:
    from tlfair.cmi_estimation import paired_cmi, require_convergence
    from tlfair.metrics import _encode_joint
    sizes = {}
    for dataset in ['adult','law']:
        directory = runs[f'real_data_{dataset}']
        metadata = json.loads((directory/'cmi_fit_metadata.json').read_text())
        cache = directory/'cmi_predictions.npz'
        path = canonical/f'{dataset}_results.pkl' if canonical else directory/'inference.pkl'
        with path.open('rb') as stream:
            expected = pickle.load(stream)
        assert sha256(cache) == metadata['prediction_sha256'] == expected['prediction_sha256']
        loader = importlib.import_module(f'analysis.analyze_{dataset}.run')
        raw = ROOT/f'data/raw/{dataset}.csv'
        assert metadata['raw_sha256'] == sha256(raw)
        assert metadata['preprocessing_sha256'] == sha256(Path(loader.__file__))
        xtr,xte,_,yte,_,gte = loader.load_data(raw, metadata['seed'])
        assert metadata['train_rows'] == len(xtr) and metadata['evaluation_rows'] == len(xte)
        with np.load(cache) as saved:
            for key,value in [('train_index',xtr.index),('test_index',xte.index),('outcomes',yte),('groups',gte)]:
                np.testing.assert_array_equal(saved[key],value)
            rebuilt = paired_cmi(saved['initial_raw_q'], _encode_joint(gte,yte))
            np.testing.assert_allclose(saved['initial_q'],rebuilt['diagnostics']['initial_q'],atol=1e-14)
            np.testing.assert_allclose(saved['targeted_q'],rebuilt['diagnostics']['q_star'],atol=1e-12)
        assert metadata['diagnostics']['converged'] and rebuilt['diagnostics']['converged']
        for method,(estimate,interval) in rebuilt['inference'].items():
            actual = expected['inference'][f'cmi_{method}']
            np.testing.assert_allclose([actual[0],*actual[1]],[estimate,*interval],atol=1e-12,rtol=1e-10)
        fit_records = pd.read_csv(directory/'cmi_diagnostics.csv')
        require_convergence(fit_records)
        validate_cmi_fit_protocol(directory, fit_records)
        assert metadata['calibration_folds'] == 0
        assert metadata['nuisance_fit']['calibration'] is None
        assert metadata['nuisance_fit']['full_training_refit'] is False
        assert fit_records.boosting_rounds.eq(metadata['nuisance_fit']['boosting_rounds']).all()
        sizes[dataset] = len(xte)
    return sizes


def validate_cmi_records() -> dict[str, int]:
    selections = json.loads((ROOT/'config/publication-runs.json').read_text())['analyses']
    runs = {name:ROOT/item['run'] for name,item in selections.items()}
    return {**validate_cmi_runs(runs, ROOT/'data/generated'),
            'real_data_evaluation_rows':validate_cmi_real_data(runs, ROOT/'data/generated')}


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
    cond=pd.read_csv(directory/'cmi_conditioning_set.csv')
    cond = cond[cond.estimator == 'tmle']
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
    matched_predictions = validate_matched_predictions()
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
                cmi_records=cmi_records,matched_evaluation_rows=matched_predictions,computation_lineage='verified',
                limitations=registry['limitations'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results/provenance/validation.json')
    args=parser.parse_args()
    result=validate(args.deep);write_json(args.output,result);print(json.dumps(result,indent=2))
