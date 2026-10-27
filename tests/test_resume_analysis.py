"""Recovery must preserve existing replicates and refuse changed computations."""
import json
from pathlib import Path
import sys

import pytest

from analysis.tools import resume_analysis
from tlfair.provenance import runtime, sha256, write_json


@pytest.fixture
def interrupted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(resume_analysis, 'ROOT', tmp_path)
    run = tmp_path/'data/runs/recovery-test/nuisance_reference'
    (run/'source').mkdir(parents=True)
    (run/'shards/cell').mkdir(parents=True)
    checkpoint = run/'shards/cell/rep_00000.csv'
    checkpoint.write_text('replicate,estimate\n0,0.1\n')
    code = "import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('estimate\\n0.1\\n')\n"
    (tmp_path/'compute.py').write_text(code)
    (run/'source/compute.py').write_text(code)
    settings = {'script':'compute.py','required':['result.csv'],'code':['compute.py']}
    write_json(tmp_path/'config/analyses.json',{'analyses':{'nuisance_reference':settings}})
    write_json(run/'run_manifest.json',{
        'analysis':'nuisance_reference','execution':'source_snapshot','status':'running',
        'settings':settings,'runtime':runtime(),'resources':{'workers':1},
        'sources':{'compute.py':sha256(tmp_path/'compute.py')},
        'code':{'compute.py':sha256(tmp_path/'compute.py')},
        'command':[sys.executable,str(run/'source/compute.py'),str(run/'result.csv')]})
    return run


def test_recovery_certifies_results_and_preserves_checkpoints(interrupted_run: Path) -> None:
    checkpoint = interrupted_run/'shards/cell/rep_00000.csv'
    before = checkpoint.read_bytes()
    resume_analysis.resume('recovery-test','nuisance_reference')
    record = json.loads((interrupted_run/'run_manifest.json').read_text())
    complete = json.loads((interrupted_run/'complete.json').read_text())
    assert checkpoint.read_bytes() == before
    assert record['status'] == record['recovery_attempts'][0]['status'] == 'complete'
    assert complete['manifest_sha256'] == sha256(interrupted_run/'run_manifest.json')
    for name,digest in record['outputs'].items():
        assert sha256(interrupted_run/name) == digest
    with pytest.raises(ValueError,match='certified complete'):
        resume_analysis.resume('recovery-test','nuisance_reference')


def test_recovery_refuses_changed_source(interrupted_run: Path) -> None:
    (interrupted_run/'source/compute.py').write_text('changed source')
    with pytest.raises(ValueError,match='Saved source changed'):
        resume_analysis.resume('recovery-test','nuisance_reference')
    assert not (interrupted_run/'result.csv').exists()
    assert not (interrupted_run/'complete.json').exists()
