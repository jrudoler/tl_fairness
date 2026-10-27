"""Checks for paired CMI provenance, completion guards and portable compilation."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tlfair.cmi_sim import paired_cmi_sim
from tlfair.provenance import sha256,write_json


def test_paired_simulation_uses_exact_figure2_fit_without_calibration():
    from sklearn.ensemble import GradientBoostingClassifier
    from experiments.nuisance_early_stopping import BOOST
    from tlfair.cmi_estimation import paired_cmi, replicate_records
    from tlfair.metrics import _encode_joint
    rng = np.random.default_rng(37)
    x = rng.normal(size=(500, 3))
    shared = 2*rng.uniform(size=500)
    logits = 1/(1+np.exp(-x@np.ones(3)))
    y = ((shared+rng.uniform(size=500)+logits)/4 > .5).astype(np.int8)
    g = ((shared+rng.uniform(size=500)+logits)/4 > .5).astype(np.int8)
    seed = int(rng.integers(0, 2**31-1))
    expected_model = GradientBoostingClassifier(**BOOST, random_state=seed).fit(
        x[:250], _encode_joint(g[:250], y[:250]))
    expected_q = expected_model.predict_proba(x[250:])
    expected = paired_cmi(expected_q, _encode_joint(g[250:], y[250:]))
    actual = paired_cmi_sim(500, 2, rng=np.random.default_rng(37))
    np.testing.assert_allclose(actual['diagnostics']['initial_q'],
                               expected['diagnostics']['initial_q'], atol=1e-12)
    for method in expected['inference']:
        estimate, interval = actual['inference'][method]
        expected_estimate, expected_interval = expected['inference'][method]
        np.testing.assert_allclose([estimate, *interval], [expected_estimate, *expected_interval], atol=1e-12)
    assert actual['diagnostics']['converged']
    assert actual['nuisance_fit']['calibration'] is None
    assert actual['nuisance_fit']['full_training_refit'] is False
    for record in replicate_records(actual):
        assert record['boosting_rounds'] == expected_model.n_estimators_
        assert record['tree_count'] == 4*expected_model.n_estimators_
        assert record['model_seed'] == seed


@pytest.mark.parametrize('plan_name', ['cmi-tmle-publication.json', 'variance-publication.json'])
def test_cmi_completion_refuses_changed_manuscript(tmp_path: Path, plan_name: str):
    from analysis.tools.complete_cmi_tmle import complete
    manuscript = tmp_path/'manuscript.tex'
    manuscript.write_text('approved prose')
    plan_path = Path('config')/plan_name
    write_json(tmp_path/plan_path,{'approved_sources':{'manuscript.tex':sha256(manuscript)}})
    manuscript.write_text('later user edits')
    with pytest.raises(ValueError,match='changed while jobs queued'):
        complete(tmp_path, plan_path)
    assert manuscript.read_text() == 'later user edits'


def test_compilation_falls_back_when_compute_node_lacks_perl_module(tmp_path: Path,monkeypatch: pytest.MonkeyPatch):
    from analysis.tools import compile_paper as compiler
    paper=tmp_path/'paper';paper.mkdir()
    (paper/'main.log').write_text('Cross references settled')
    commands=[]

    def execute(command,**kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command,1 if command[0]=='perl' else 0)

    monkeypatch.setattr(compiler.subprocess,'run',execute)
    monkeypatch.delenv('XDG_CACHE_HOME',raising=False)
    monkeypatch.delenv('MPLCONFIGDIR',raising=False)
    compiler.compile_manuscript(tmp_path)
    assert [command[0] for command in commands] == ['perl','xelatex','bibtex','xelatex']
    assert (tmp_path/'logs/cache').is_dir()


def test_cmi_completion_resumes_after_selection_without_selecting_again(tmp_path: Path,monkeypatch: pytest.MonkeyPatch):
    from analysis.tools import complete_cmi_tmle as completion
    write_json(tmp_path/'config/cmi-tmle-publication.json',{
        'approved_sources':{},'run_id':'fresh','publication_id':'selected','analyses':['cmi_estimation']})
    write_json(tmp_path/'config/publication.json',{'publication_id':'selected'})
    write_json(tmp_path/'config/publication-runs.json',{'analyses':{
        'cmi_estimation':{'run':'data/runs/fresh/cmi_estimation'}}})
    monkeypatch.setattr(completion,'validate_cmi_runs',lambda *args:{})
    monkeypatch.setattr(completion,'validate_cmi_real_data',lambda *args:{})
    monkeypatch.setattr(completion,'audit',lambda *args:{'status':'verified'})

    def reject_selection(*args,**kwargs):
        raise AssertionError('An already selected publication must not be selected again')

    monkeypatch.setattr(completion,'select',reject_selection)

    def failed_build(command,**kwargs):
        raise subprocess.CalledProcessError(1,command)

    monkeypatch.setattr(completion.subprocess,'run',failed_build)
    with pytest.raises(subprocess.CalledProcessError):
        completion.complete(tmp_path)
    assert not (tmp_path/'results/provenance/completion-selected.json').exists()
    (tmp_path/'paper').mkdir()
    (tmp_path/'paper/main.pdf').write_bytes(b'%PDF-test')
    monkeypatch.setattr(completion.subprocess,'run',lambda command,**kwargs:subprocess.CompletedProcess(command,0))
    result=completion.complete(tmp_path)
    assert result['resumed'] and result['build']=='passed'
