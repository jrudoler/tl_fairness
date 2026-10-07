"""Validate a complete matched experiment before publishing figures or tables."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from experiments.nuisance_training import SCENARIOS, SIZES, CONTROL_SIZES, CASES, METHODS, KEYS, summarize


def validate(directory):
    directory = Path(directory)
    raw = pd.read_csv(directory/'replicates.csv.gz')
    summary = pd.read_csv(directory/'summary.csv')
    population = pd.read_csv(directory/'population.csv')
    pairs = pd.read_csv(directory/'paired_comparisons.csv')
    manifest = json.loads((directory/'manifest.json').read_text())
    has_cv = 'cross_validated' in set(raw.case)
    has_matched_rate = 'matched_rate_shards' in manifest
    base_manifest = manifest.get('base_manifest', manifest)
    assert base_manifest['expected_reps'] == 1000
    assert raw.test_size.eq(2000).all()
    assert len(raw) == (132000 if has_cv else 118000)
    assert not raw.duplicated(KEYS+['replicate']).any()
    assert np.isfinite(raw[['estimate', 'ci_low', 'ci_high', 'se', 'width', 'truth']]).all().all()
    assert (raw.ci_low <= raw.estimate).all() and (raw.estimate <= raw.ci_high).all()
    np.testing.assert_allclose(raw.width, 3.92*raw.se)
    for (scenario, n), sub in raw.groupby(['scenario', 'train_size']):
        expected_cases = set(CASES) if scenario == 'primary' and n in CONTROL_SIZES else {'default', 'higher_capacity'}
        if has_cv and scenario == 'primary':
            expected_cases.add('cross_validated')
        assert set(sub.case) == expected_cases
        assert sub.groupby(['case','method']).replicate.nunique().eq(1000).all()
        assert sub.groupby('replicate').data_hash.nunique().eq(1).all()
        assert sub.groupby(['replicate','outcome_model']).outcome_hash.nunique().eq(1).all()
        assert sub.groupby(['replicate','group_model']).group_hash.nunique().eq(1).all()
        assert sub.groupby(['replicate','case']).method.nunique().eq(2).all()
    assert set(raw[['scenario','train_size']].itertuples(index=False,name=None)) == {
        (s,n) for s in SCENARIOS for n in (SIZES if s=='primary' else CONTROL_SIZES)}
    _, rebuilt, paired = summarize(raw, population)
    pd.testing.assert_frame_equal(summary, rebuilt, check_dtype=False, atol=1e-12, rtol=1e-10)
    pd.testing.assert_frame_equal(pairs, paired, check_dtype=False, atol=1e-12, rtol=1e-10)
    for scenario, sub in summary.groupby('scenario'):
        integration_se = population.set_index('scenario').loc[scenario, 'integration_se']
        assert integration_se < .01*sub.bias_mcse.min()
    signatures = set()
    for metadata in base_manifest['shards'].values():
        signatures.add(json.dumps({k: metadata[k] for k in ['seed','source_hashes','python','numpy','scipy','sklearn',
                                                             'default_learner','higher_capacity_learner','primary_parameters']}, sort_keys=True))
        assert metadata['test_size'] == 2000
    assert len(signatures) == 1, 'Mixed source, seed, model settings, or runtime versions'
    if 'early_stopping' in manifest:
        from experiments.nuisance_early_stopping import BOOST, HIGH, EARLY_STOPPING
        assert manifest['early_stopping'] == EARLY_STOPPING
        assert len(manifest['shards']) == 320
        original_manifest=json.loads((directory.parent/'nuisance_training'/'manifest.json').read_text())
        reference=next(iter(original_manifest['shards'].values()))
        for metadata in manifest['shards'].values():
            assert metadata['default_learner'] == BOOST
            assert metadata['higher_capacity_learner'] == HIGH
            for key in ['seed','python','numpy','scipy','sklearn']:
                assert metadata[key] == reference[key]
            for key,value in reference['source_hashes'].items():
                assert metadata['source_hashes'][key] == value
        for role in ['outcome','group']:
            for name,cap in [('default',100),('higher_capacity',1000)]:
                counts=raw.loc[raw[role+'_model']==name,role+'_trees']
                assert counts.between(1,cap).all() and counts.eq(counts.astype(int)).all()
            assert np.isfinite(raw[role+'_probability_mse']).all()
            assert raw[role+'_probability_mse'].ge(0).all()
        diagnostics=pd.read_csv(directory/'fit_diagnostics.csv.gz')
        expected=raw.drop_duplicates(['scenario','train_size','replicate','outcome_model','group_model'])
        pd.testing.assert_frame_equal(diagnostics,expected.reset_index(drop=True),check_dtype=False,atol=1e-12)
        original=pd.read_csv(directory.parent/'nuisance_training'/'replicates.csv.gz')
        from analysis.nuisance_tables.control_audit import audit_unchanged
        audit=audit_unchanged(raw,original)
        assert audit==json.loads((directory/'control_audit.json').read_text())
    if has_matched_rate:
        from experiments.nuisance_matched_rate import HIGH
        assert manifest['replacement_case'] == 'higher_capacity'
        assert manifest['higher_capacity_learner'] == HIGH
        assert len(manifest['matched_rate_shards']) == 640
        reference = next(iter(base_manifest['shards'].values()))
        replacement_sources = set()
        for metadata in manifest['matched_rate_shards'].values():
            assert metadata['higher_capacity_learner'] == HIGH
            assert metadata['default_learner'] == reference['default_learner']
            assert metadata['seed'] == reference['seed'] and metadata['test_size'] == 2000
            for key in ['python','numpy','scipy','sklearn']:
                assert metadata[key] == reference[key]
            for key,value in reference['source_hashes'].items():
                assert metadata['source_hashes'][key] == value
            replacement_sources.add(metadata['source_hashes']['experiments/nuisance_matched_rate.py'])
        assert len(replacement_sources) == 1
        original = pd.read_csv(directory.parent/'nuisance_training'/'replicates.csv.gz')
        identity = KEYS+['replicate']
        retained = raw[raw.case!='higher_capacity'].sort_values(identity).reset_index(drop=True)
        previous = original[original.case!='higher_capacity'].sort_values(identity).reset_index(drop=True)
        pd.testing.assert_frame_equal(retained, previous, check_dtype=False, atol=1e-14, rtol=1e-12)
        comparisons = pd.read_csv(directory/'rate_comparisons.csv')
        assert len(comparisons) == 32
        for record in comparisons.itertuples(index=False):
            select = lambda df: df[(df.scenario==record.scenario)&(df.train_size==record.train_size)
                                   &(df.method==record.method)&(df.case=='higher_capacity')].set_index('replicate')
            a,b = select(raw),select(original)
            b = b.loc[a.index]
            assert a.data_hash.eq(b.data_hash).all()
            for name,column in [('coverage','covered'),('bias','error')]:
                differences = a[column].astype(float)-b[column].astype(float)
                np.testing.assert_allclose(getattr(record,name+'_difference'),differences.mean(),atol=1e-12)
                np.testing.assert_allclose(getattr(record,name+'_difference_mcse'),differences.sem(),atol=1e-12)
    if has_cv:
        from experiments.nuisance_tuning import GRID, CV_FOLDS
        tuning_sources = set()
        assert len(manifest['tuning_shards']) == 280
        for metadata in manifest['tuning_shards'].values():
            tuning_sources.add(metadata['source_hashes']['experiments/nuisance_tuning.py'])
            assert metadata['grid'] == GRID and metadata['cv_folds'] == CV_FOLDS
            assert metadata['seed'] == 20261005 and metadata['test_size'] == 2000
            reference = next(iter(base_manifest['shards'].values()))
            for key in ['python','numpy','scipy','sklearn']:
                assert metadata[key] == reference[key]
            for key,value in reference['source_hashes'].items():
                assert metadata['source_hashes'][key] == value
        assert len(tuning_sources) == 1, 'Mixed tuning implementations'
        selections = pd.read_csv(directory/'tuning_selections.csv')
        assert len(selections) == 14000
        assert not selections.duplicated(['train_size','replicate','role']).any()
        assert set(selections.role) == {'outcome','group'}
        candidates = {(p['n_estimators'],p['max_depth'],p['learning_rate']) for p in GRID}
        assert set(selections[['n_estimators','max_depth','learning_rate']].itertuples(index=False,name=None)) <= candidates
        from experiments.nuisance_training import BOOST, HIGH
        matched = raw[(raw.scenario=='primary') & (raw.method=='TL data fairness')]
        tuned = matched[matched.case=='cross_validated'].set_index(['train_size','replicate'])
        for role in ['outcome','group']:
            choice = selections[selections.role==role]
            for case,params in [('default',BOOST),('higher_capacity',HIGH)]:
                same = choice[(choice.n_estimators==params['n_estimators']) &
                              (choice.max_depth==params['max_depth']) &
                              (choice.learning_rate==params['learning_rate'])]
                ids = same.set_index(['train_size','replicate']).index
                baseline = matched[matched.case==case].set_index(['train_size','replicate'])
                assert tuned.loc[ids,role+'_hash'].eq(baseline.loc[ids,role+'_hash']).all()
        comparisons = pd.read_csv(directory/'learner_comparisons.csv')
        assert len(comparisons) == 28
        for record in comparisons.itertuples(index=False):
            part = raw[(raw.scenario=='primary') & (raw.train_size==record.train_size) & (raw.method==record.method)]
            tuned = part[part.case=='cross_validated'].set_index('replicate')
            other = record.comparison.removeprefix('cross_validated minus ')
            baseline = part[part.case==other].set_index('replicate').loc[tuned.index]
            for name,column in [('coverage','covered'),('bias','error')]:
                differences = tuned[column].astype(float)-baseline[column].astype(float)
                np.testing.assert_allclose(getattr(record,name+'_difference'),differences.mean(),atol=1e-12)
                np.testing.assert_allclose(getattr(record,name+'_difference_mcse'),differences.sem(),atol=1e-12)
    print(f'Validated {len(raw):,} paired-method rows, {len(summary)} summaries, and all 16 prespecified configurations.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', default='data/generated/nuisance_early_stopping')
    validate(p.parse_args().directory)
