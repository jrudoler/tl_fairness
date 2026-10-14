"""Scientific origin checks must reject plausible but untraceable results."""
import json
from pathlib import Path
import subprocess

import pytest
import pandas as pd
import numpy as np

from analysis.tools.audit_publication_runs import audit
from tlfair.provenance import runtime, sha256, write_json


@pytest.fixture
def recorded_run(tmp_path: Path) -> Path:
    (tmp_path/'config').mkdir()
    (tmp_path/'compute.py').write_text('original computation\n')
    settings = {'script': 'compute.py', 'code': ['compute.py'], 'parameters': {'seed': 1}, 'required': ['result.csv']}
    write_json(tmp_path/'config/analyses.json', {'rng_version': 'test-v1', 'analyses': {'test': settings}})
    write_json(tmp_path/'config/publication-runs.json', {'analyses': {
        'test': {'run': 'data/runs/test/test', 'inputs': {'data/generated/result.csv': 'result.csv'}}}})
    write_json(tmp_path/'config/artifacts.json', {'artifacts': [{'dependencies': ['data/generated/result.csv']}]})
    directory = tmp_path/'data/runs/test/test'
    (directory/'source').mkdir(parents=True)
    (directory/'source/compute.py').write_bytes((tmp_path/'compute.py').read_bytes())
    (directory/'result.csv').write_text('estimate\n0.123\n')
    (tmp_path/'data/generated').mkdir(parents=True)
    (tmp_path/'data/generated/result.csv').write_bytes((directory/'result.csv').read_bytes())
    digest = sha256(tmp_path/'compute.py')
    record = {'execution': 'source_snapshot', 'analysis': 'test', 'status': 'complete', 'settings': settings,
              'rng_version': 'test-v1', 'code': {'compute.py': digest}, 'sources': {'compute.py': digest},
              'outputs': {'result.csv': sha256(directory/'result.csv')}, 'runtime': runtime()}
    write_json(directory/'run_manifest.json', record)
    write_json(directory/'complete.json', {'status': 'complete', 'manifest_sha256': sha256(directory/'run_manifest.json')})
    return tmp_path


def test_matching_completed_run_verifies(recorded_run: Path) -> None:
    assert audit(recorded_run)['status'] == 'verified'


def test_selection_preserves_old_inputs_and_tracks_completion_records(recorded_run: Path) -> None:
    from analysis.tools.select_publication_runs import select
    published = recorded_run/'data/generated/result.csv'
    write_json(recorded_run/'config/publication.json', {
        'publication_id':'previous', 'inputs':[{'path':'data/generated/result.csv','sha256':sha256(published)}]})
    old_bytes = published.read_bytes()
    directory = recorded_run/'data/runs/test/test'
    (directory/'result.csv').write_text('estimate\n0.456\n')
    record = json.loads((directory/'run_manifest.json').read_text())
    record['outputs']['result.csv'] = sha256(directory/'result.csv')
    write_json(directory/'run_manifest.json',record)
    write_json(directory/'complete.json',{'status':'complete','manifest_sha256':sha256(directory/'run_manifest.json')})
    result = select('test','new-selection',recorded_run)
    assert result['status'] == 'verified'
    assert (recorded_run/result['previous_inputs']/'data/generated/result.csv').read_bytes() == old_bytes
    assert published.read_bytes() == (directory/'result.csv').read_bytes()
    selection = json.loads((recorded_run/'config/publication-runs.json').read_text())['analyses']['test']
    assert (recorded_run/selection['manifest']).is_file()
    assert (recorded_run/selection['complete']).is_file()
    assert audit(recorded_run)['status'] == 'verified'
    (recorded_run/selection['manifest']).write_text('{}')
    assert audit(recorded_run)['status'] == 'unverified'


def test_selection_rejects_changed_code_before_replacing_any_inputs(recorded_run: Path) -> None:
    from analysis.tools.select_publication_runs import select
    published = recorded_run/'data/generated/result.csv'
    write_json(recorded_run/'config/publication.json', {
        'inputs':[{'path':'data/generated/result.csv','sha256':sha256(published)}]})
    original = published.read_bytes()
    (recorded_run/'compute.py').write_text('changed computation')
    with pytest.raises(ValueError,match='preflight'):
        select('test','new-selection',recorded_run)
    assert published.read_bytes() == original
    assert not (recorded_run/'data/imported/publication-before-new-selection').exists()


def test_select_additional_analysis_retains_existing_run(recorded_run: Path) -> None:
    import shutil
    from analysis.tools.select_publication_runs import select
    published = recorded_run/'data/generated/result.csv'
    write_json(recorded_run/'config/publication.json', {
        'inputs':[{'path':'data/generated/result.csv','sha256':sha256(published)}]})
    specifications = json.loads((recorded_run/'config/analyses.json').read_text())
    specifications['analyses']['added'] = specifications['analyses']['test']
    write_json(recorded_run/'config/analyses.json', specifications)
    directory = recorded_run/'data/runs/new/added'
    shutil.copytree(recorded_run/'data/runs/test/test', directory)
    record = json.loads((directory/'run_manifest.json').read_text())
    record['analysis'] = 'added'
    write_json(directory/'run_manifest.json',record)
    write_json(directory/'complete.json', {'status':'complete','manifest_sha256':sha256(directory/'run_manifest.json')})
    select('new', 'mixed-selection', recorded_run, analyses=['added'], additional={
        'added':{'inputs':{'data/generated/added.csv':'result.csv'}}})
    selected = json.loads((recorded_run/'config/publication-runs.json').read_text())['analyses']
    assert selected['test']['run'] == 'data/runs/test/test'
    assert selected['added']['run'] == 'data/runs/new/added'
    assert audit(recorded_run)['status'] == 'verified'


def test_queued_completion_refuses_to_overwrite_later_manuscript_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis.tools import complete_matched_real_data as completion
    monkeypatch.setattr(completion, 'ROOT', tmp_path)
    (tmp_path/'config').mkdir()
    manuscript, draft = tmp_path/'manuscript.tex', tmp_path/'draft.tex'
    manuscript.write_text('original manuscript')
    draft.write_text('approved changes')
    write_json(tmp_path/'config/matched-real-data-publication.json', {
        'manuscript_source':'manuscript.tex', 'manuscript_draft':'draft.tex',
        'manuscript_before_sha256':sha256(manuscript), 'manuscript_draft_sha256':sha256(draft)})
    manuscript.write_text('later user edits')
    with pytest.raises(ValueError, match='changed while jobs were queued'):
        completion.main()
    assert manuscript.read_text() == 'later user edits'


@pytest.mark.parametrize('path,issue', [
    ('compute.py', 'Current computation dependency differs'),
    ('data/runs/test/test/source/compute.py', 'Saved computation source differs'),
    ('data/generated/result.csv', 'Publication input is not the recorded run output'),
    ('data/runs/test/test/result.csv', 'Recorded run output differs'),
    ('data/runs/test/test/run_manifest.json', 'Completion record does not certify'),
])
def test_tampering_breaks_the_origin_chain(recorded_run: Path, path: str, issue: str) -> None:
    target = recorded_run/path
    if target.suffix == '.json':
        record = json.loads(target.read_text())
        record['run_id'] = 'changed-after-completion'
        write_json(target, record)
    else:
        target.write_text('plausible replacement\n')
    result = audit(recorded_run)
    assert result['status'] == 'unverified'
    assert any(issue in message for message in result['analyses']['test']['issues'])


def test_unmapped_manuscript_data_cannot_pass(recorded_run: Path) -> None:
    write_json(recorded_run/'config/artifacts.json', {'artifacts': [{'dependencies': ['data/generated/unmapped.csv']}]})
    assert audit(recorded_run)['issues']


def test_runner_executes_saved_source_even_if_checkout_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import analysis.tools.run_analysis as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'git_state', lambda: {'commit': 'test'})
    (tmp_path/'config').mkdir()
    (tmp_path/'clean-figs.mplstyle').write_text('')
    (tmp_path/'uv.lock').write_text('test lock')
    (tmp_path/'pyproject.toml').write_text('')
    script = tmp_path/'compute.py'
    script.write_text('import argparse\nfrom pathlib import Path\np=argparse.ArgumentParser()\np.add_argument("--n-jobs")\np.add_argument("--output")\na=p.parse_args()\nPath(a.output).write_text("saved code")\n')
    config = tmp_path/'config/analyses.json'
    write_json(config, {'rng_version': 'test-v1', 'analyses': {'test': {
        'script': 'compute.py', 'code': ['compute.py'], 'parameters': {},
        'outputs': {'output': 'result.txt'}, 'required': ['result.txt']}}})
    actual_run = subprocess.run

    def change_live_checkout_then_execute(*args, **kwargs):
        script.write_text('raise RuntimeError("Live checkout must not execute")\n')
        return actual_run(*args, **kwargs)

    monkeypatch.setattr(runner.subprocess, 'run', change_live_checkout_then_execute)
    output = runner.run('test', 'test', config)
    assert (output/'result.txt').read_text() == 'saved code'
    record = json.loads((output/'run_manifest.json').read_text())
    assert record['execution'] == 'source_snapshot'
    assert record['code']['compute.py'] == sha256(output/'source/compute.py')


def test_cmi_tail_diagnostics_explain_bad_coverage_without_positive_rejection() -> None:
    from analysis.tools.compare_reproduction import null_diagnostics
    frame = pd.DataFrame({'sample_size':[100]*3,'estimate':[-1.,0.,1.],
                          'ci_low':[-1.1,-.1,.9],'ci_high':[-.9,.1,1.1]})
    row = null_diagnostics(frame, ['sample_size']).iloc[0]
    assert row.two_sided_coverage == pytest.approx(1/3)
    assert row.two_sided_positive_exclusion == pytest.approx(1/3)
    assert row.two_sided_negative_exclusion == pytest.approx(1/3)
    negative = null_diagnostics(frame.iloc[:1], ['sample_size']).iloc[0]
    assert negative.two_sided_coverage == 0
    assert negative.two_sided_negative_exclusion == 1
    assert negative.one_sided_positive_rejection == 0


def test_comparison_aligns_grid_labels_instead_of_csv_row_order(tmp_path: Path) -> None:
    from analysis.tools.compare_reproduction import compare_csv
    old, new = tmp_path/'old.csv', tmp_path/'new.csv'
    pd.DataFrame({'sample_size':[100,200],'kappa':[0.,0.],'coverage':[.9,.8]}).to_csv(old,index=False)
    pd.DataFrame({'sample_size':[200,100],'kappa':[0.,0.],'coverage':[.85,.95]}).to_csv(new,index=False)
    result = compare_csv(old,new,'cmi_estimation')
    np.testing.assert_allclose(result.difference, .05)
    pd.DataFrame({'sample_size':[300],'kappa':[0.],'coverage':[.9]}).to_csv(new,index=False)
    with pytest.raises(ValueError,match='grids differ'):
        compare_csv(old,new,'cmi_estimation')


def test_cmi_comparison_excludes_retired_method_but_requires_active_grid(tmp_path: Path) -> None:
    from analysis.tools.compare_reproduction import compare_csv
    old, new = tmp_path/'cmi_estimation_comparison.csv', tmp_path/'comparison.csv'
    pd.DataFrame({'sample size':[100]*3,'kappa':[0.]*3,
                  'type':['TL','KNN','TL-sep'],'mean':[.1,.2,-.3]}).to_csv(old,index=False)
    pd.DataFrame({'sample size':[100]*2,'kappa':[0.]*2,
                  'type':['TL','KNN'],'mean':[.15,.25]}).to_csv(new,index=False)
    result = compare_csv(old,new,'cmi_estimation')
    assert set(result['type']) == {'TL','KNN'}
    np.testing.assert_allclose(result.difference, .05)
    pd.read_csv(new).iloc[:1].to_csv(new,index=False)
    with pytest.raises(ValueError,match='grids differ'):
        compare_csv(old,new,'cmi_estimation')
