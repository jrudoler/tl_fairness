import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tlfair.cmi_sim import cmi_coverage_sim,cmi_compare
from analysis.tools.validate_publication import verify_inputs


def score(n,kappa,rng):
    value=rng.normal()+kappa
    if value < -1:
        raise ValueError('Exercise deterministic retry')
    return value,(value-.2,value+.2)


def test_cmi_coverage_streams_and_records_are_worker_invariant():
    records=[];parallel=[]
    a=cmi_coverage_sim(100,.5,.5,sims=12,fn=score,rng=np.random.default_rng(3),n_jobs=1,records=records)
    b=cmi_coverage_sim(100,.5,.5,sims=12,fn=score,rng=np.random.default_rng(3),n_jobs=2,records=parallel)
    assert a==b and records==parallel
    assert a[0]==np.mean([r['covered'] for r in records])
    assert a[1]==pytest.approx(np.mean([r['estimate']-.5 for r in records]))


def test_cmi_comparison_worker_invariant():
    records=[];parallel=[]
    a=cmi_compare(80,repeats=2,params=[0,.5],rng=np.random.default_rng(87),n_jobs=1,records=records)
    b=cmi_compare(80,repeats=2,params=[0,.5],rng=np.random.default_rng(87),n_jobs=2,records=parallel)
    pd.testing.assert_frame_equal(a,b)
    assert records==parallel


def test_publication_registry_rejects_modified_or_missing_input(tmp_path):
    p=tmp_path/'result.csv';p.write_text('valid original result')
    registry={'inputs':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]}
    verify_inputs(registry,tmp_path)
    p.write_text('plausible but stale result')
    with pytest.raises(ValueError,match='checksum mismatch'):verify_inputs(registry,tmp_path)
    p.unlink()
    with pytest.raises(FileNotFoundError,match='restore the recorded bundle'):verify_inputs(registry,tmp_path)


def test_new_run_cannot_escape_or_overwrite_run_directory(tmp_path,monkeypatch):
    import analysis.tools.run_analysis as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    config=tmp_path/'config.json';config.write_text(json.dumps({'analyses':{'example':{}}}))
    with pytest.raises(ValueError,match='simple directory'):runner.run('example','../publication',config)
    existing=tmp_path/'data/runs/test/example';existing.mkdir(parents=True)
    with pytest.raises(FileExistsError,match='refusing to overwrite'):runner.run('example','test',config)


def test_publication_dag_has_no_simulation_rules():
    text=Path('Snakefile').read_text()
    import re
    assert not any(name.startswith(('sim_','analyze_','download_'))
                   for name in re.findall(r'^rule (\w+):',text,re.M))
    assert 'run_analysis.py' not in text


def test_retraining_plot_only_leaves_input_bundle_unchanged(tmp_path, monkeypatch):
    import sys
    import experiments.parity_retraining as experiment
    original = tmp_path / 'run.log'
    original.write_text('original simulation log\n')
    summary = tmp_path / 'summary.csv'
    summary.write_text('effect,coverage_data\n0,0.95\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    handlers = []
    monkeypatch.setattr(experiment.logging, 'basicConfig', lambda **kwargs: handlers.extend(kwargs['handlers']))
    monkeypatch.setattr(experiment, 'plot', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['parity_retraining', '--plot-only', '--output-dir', str(tmp_path)])
    experiment.main()
    assert all(not isinstance(h, experiment.logging.FileHandler) for h in handlers)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
