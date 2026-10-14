"""Submit comparison-only reruns; record each job and preserve publication inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tlfair.provenance import git_state, sha256, write_json

LOG = logging.getLogger(__name__)
# CPU analyses only. These allocations reflect the full configured grids,
# including the much slower fixed-tree nuisance reference.
RESOURCES = {
    'parity_retraining': ('12:00:00', '32G'),
    'cmi_estimation': ('24:00:00', '64G'),
    'cmi_conditioning_set': ('12:00:00', '32G'),
    'real_data_adult': ('24:00:00', '32G'),
    'real_data_law': ('12:00:00', '32G'),
    'nuisance_reference': ('2-00:00:00', '32G'),
    'nuisance_early_stopping': ('2-00:00:00', '32G'),
    'model_fairness_adult': ('12:00:00', '64G'),
    'model_fairness_law': ('04:00:00', '32G'),
}


def submit(run_id: str, dry_run: bool = False) -> dict[str, Any]:
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', run_id):
        raise ValueError('run-id must be a simple directory name')
    record_path = ROOT/'results/provenance'/f'submission-{run_id}.json'
    if record_path.exists() or (ROOT/'data/runs'/run_id).exists():
        raise FileExistsError('Use a fresh run ID; a submission or run already exists')
    (ROOT/'logs/slurm').mkdir(parents=True, exist_ok=True)
    specs = json.loads((ROOT/'config/analyses.json').read_text())['analyses']
    if set(specs) != set(RESOURCES):
        raise ValueError('Update the resource plan when the configured analyses change')
    record = {'run_id':run_id,'submitted_at':datetime.now(timezone.utc).isoformat(),
              'git':git_state(),'analyses_sha256':sha256(ROOT/'config/analyses.json'),
              'publication_sha256':sha256(ROOT/'config/publication.json'),
              'jobs':{},'publication_changed':False}

    def dispatch(name: str, arguments: list[str]) -> str:
        LOG.info('Submitting %s: %s', name, shlex.join(arguments))
        if dry_run:
            return str(900000+len(record['jobs']))
        result = subprocess.run(arguments,cwd=ROOT,check=True,text=True,capture_output=True)
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise ValueError(f'Unexpected sbatch response: {result.stdout}')
        return job_id

    for analysis, (time_limit, memory) in RESOURCES.items():
        command = ['sbatch','--parsable','--partition=whartonstat',f'--chdir={ROOT}',
                   f'--job-name=fairness-{analysis.replace("_","-")}',
                   '--cpus-per-task=8',f'--mem={memory}',f'--time={time_limit}',
                   '--output=logs/slurm/%x-%j.log','--error=logs/slurm/%x-%j.log']
        if specs[analysis].get('requires_reference'):
            command += [f'--dependency=afterok:{record["jobs"]["nuisance_reference"]["id"]}',
                        '--kill-on-invalid-dep=yes']
        command += ['slurm/analysis.sbatch',analysis,run_id]
        if specs[analysis].get('requires_reference'):
            command += [str(ROOT/'data/runs'/run_id/'nuisance_reference')]
        job_id = dispatch(analysis,command)
        record['jobs'][analysis] = {'id':job_id,'command':command}
        if not dry_run:
            write_json(record_path,record)
    dependencies = ':'.join(job['id'] for job in record['jobs'].values())
    comparison = ['sbatch','--parsable','--partition=whartonstat',f'--chdir={ROOT}',
                  '--job-name=fairness-reproduction-comparison','--cpus-per-task=1','--mem=16G','--time=01:00:00',
                  '--output=logs/slurm/%x-%j.log','--error=logs/slurm/%x-%j.log',
                  f'--dependency=afterany:{dependencies}','--kill-on-invalid-dep=yes',
                  '--wrap='+shlex.join([str(ROOT/'.venv/bin/python'),
                      'analysis/tools/compare_reproduction.py','--run-id',run_id])]
    job_id = dispatch('comparison',comparison)
    record['jobs']['comparison'] = {'id':job_id,'command':comparison}
    if not dry_run:
        write_json(record_path,record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    record = submit(args.run_id,args.dry_run)
    LOG.info('Job IDs: %s', {name:job['id'] for name,job in record['jobs'].items()})


if __name__ == '__main__':
    main()
