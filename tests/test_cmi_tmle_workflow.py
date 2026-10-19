"""Checks for paired CMI provenance, completion guards and portable compilation."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tlfair.cmi_sim import cmi_sim, paired_cmi_sim
from tlfair.provenance import sha256,write_json


def test_paired_simulation_preserves_original_one_step_draw_and_fit():
    expected = cmi_sim(500,2,rng=np.random.default_rng(37))
    actual = paired_cmi_sim(500,2,rng=np.random.default_rng(37))
    estimate,interval = actual['inference']['one_step']
    np.testing.assert_allclose([estimate,*interval],[expected[0],*expected[1]],atol=1e-12)
    assert actual['diagnostics']['converged']


def test_cmi_completion_refuses_changed_manuscript(tmp_path: Path):
    from analysis.tools.complete_cmi_tmle import complete
    manuscript = tmp_path/'manuscript.tex'
    manuscript.write_text('approved prose')
    write_json(tmp_path/'config/cmi-tmle-publication.json',{'approved_sources':{'manuscript.tex':sha256(manuscript)}})
    manuscript.write_text('later user edits')
    with pytest.raises(ValueError,match='changed while jobs queued'):
        complete(tmp_path)
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
