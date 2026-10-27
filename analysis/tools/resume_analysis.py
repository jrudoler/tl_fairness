"""Resume an interrupted nuisance run from its certified source and saved fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tlfair.provenance import runtime, sha256, write_json

LOG = logging.getLogger(__name__)


def resume(run_id: str, analysis: str) -> None:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', run_id):
        raise ValueError('Use a simple run ID')
    directory = ROOT/'data/runs'/run_id/analysis
    manifest = directory/'run_manifest.json'
    with (directory/'resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(manifest.read_text())
        if (directory/'complete.json').exists() or record['status'] == 'complete':
            raise ValueError('Refusing to resume a certified complete run')
        if record['analysis'] != analysis or record['execution'] != 'source_snapshot':
            raise ValueError('Unexpected analysis or execution provenance')
        settings = json.loads((ROOT/'config/analyses.json').read_text())['analyses'][analysis]
        if record['settings'] != settings:
            raise ValueError('Scientific settings changed')
        current = runtime()
        if any(current.get(k) != v for k,v in record['runtime'].items() if k != 'platform'):
            raise ValueError('Runtime dependencies changed')
        snapshot = directory/'source'

        def verify_sources() -> None:
            for path, expected in record['sources'].items():
                if sha256(snapshot/path) != expected:
                    raise ValueError(f'Saved source changed: {path}')
            for path, expected in record['code'].items():
                if sha256(ROOT/path) != expected:
                    raise ValueError(f'Current computation code changed: {path}')
            for path, expected in record.get('input_files', {}).items():
                if sha256(ROOT/path) != expected:
                    raise ValueError(f'Raw input changed: {path}')
            for path, expected in record.get('reference', {}).items():
                if sha256(path) != expected:
                    raise ValueError(f'Nuisance reference changed: {path}')

        verify_sources()
        command = record['command']
        if Path(command[1]).resolve() != (snapshot/settings['script']).resolve():
            raise ValueError('Recorded command does not execute the saved analysis source')
        checkpoints = {str(p.relative_to(directory)):sha256(p)
                       for p in sorted((directory/'shards').rglob('*'))
                       if p.is_file() and (p.name.startswith('rep_') and p.suffix == '.csv'
                                          or p.name == 'metadata.json')}
        recovery = directory/'recovery'/f"attempt-{len(record.get('recovery_attempts',[]))+1:02d}"
        recovery.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest, recovery/'previous_manifest.json')
        shutil.copy2(__file__, recovery/'resume_analysis.py')
        write_json(recovery/'checkpoints.json', checkpoints)
        attempt = {'started':datetime.now(timezone.utc).isoformat(),
                   'reason':'Recover saved replicates after home-directory quota failure',
                   'driver_sha256':sha256(recovery/'resume_analysis.py'),
                   'checkpoint_manifest_sha256':sha256(recovery/'checkpoints.json'),
                   'recovery_directory':str(recovery.relative_to(directory)),
                   'previous_manifest_sha256':sha256(recovery/'previous_manifest.json'),
                   'command':command, 'status':'running'}
        record.setdefault('recovery_attempts',[]).append(attempt)
        record['status'] = 'running'
        record.pop('error',None)
        write_json(manifest, record)
        try:
            LOG.info('Resuming %s with %d preserved checkpoint files',analysis,len(checkpoints))
            with (recovery/'execution.log').open('w') as log:
                subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT,
                               check=True, env=dict(os.environ,OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                               LOKY_MAX_CPU_COUNT=str(record['resources']['workers']),
                               PYTHONPATH=str(snapshot)))
            verify_sources()
            for path,expected in checkpoints.items():
                if sha256(directory/path) != expected:
                    raise ValueError(f'Existing checkpoint changed during recovery: {path}')
            for name in settings['required']:
                if not (directory/name).is_file():
                    raise FileNotFoundError(directory/name)
            record['outputs'] = {str(p.relative_to(directory)):sha256(p)
                for p in sorted(directory.rglob('*')) if p.is_file()
                and 'source' not in p.relative_to(directory).parts and p != manifest}
            record['status'] = attempt['status'] = 'complete'
        except Exception as error:
            record['status'] = attempt['status'] = 'failed'
            record['error'] = str(error)
            raise
        finally:
            record['finished'] = attempt['finished'] = datetime.now(timezone.utc).isoformat()
            write_json(manifest, record)
        write_json(directory/'complete.json',{'status':'complete','manifest_sha256':sha256(manifest)})
        LOG.info('Completed %s; all previous checkpoint hashes preserved',analysis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--analysis',required=True,choices=['nuisance_reference','nuisance_early_stopping'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    resume(args.run_id,args.analysis)
