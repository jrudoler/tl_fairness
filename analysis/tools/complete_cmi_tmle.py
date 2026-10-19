"""Validate, select and build approved CMI TMLE runs; safe to retry a failed build."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.tools.audit_publication_runs import audit
from analysis.tools.select_publication_runs import select
from analysis.tools.validate_publication import validate_cmi_runs, validate_cmi_real_data
from tlfair.provenance import sha256, write_json

LOG = logging.getLogger(__name__)


def complete(root: Path = ROOT) -> dict:
    plan = json.loads((root/'config/cmi-tmle-publication.json').read_text())
    for path,expected in plan['approved_sources'].items():
        if sha256(root/path) != expected:
            raise ValueError(f'Approved manuscript/workflow changed while jobs queued: {path}; reconcile before completion')
    runs = {name:root/'data/runs'/plan['run_id']/name for name in plan['analyses']}
    validation = {**validate_cmi_runs(runs), 'real_data_evaluation_rows':validate_cmi_real_data(runs)}
    publication = json.loads((root/'config/publication.json').read_text())
    if publication['publication_id'] == plan['publication_id']:
        selections = json.loads((root/'config/publication-runs.json').read_text())['analyses']
        for name,path in runs.items():
            if selections[name]['run'] != str(path.relative_to(root)):
                raise ValueError('Publication ID already exists with different selected runs')
        if audit(root)['status'] != 'verified':
            raise ValueError('Previously selected CMI publication no longer verifies')
        selection = {'publication_id':plan['publication_id'],'run_id':plan['run_id'],'status':'verified','resumed':True}
    else:
        if sha256(root/'config/publication.json') != plan['previous_registry_sha256']:
            raise ValueError('Previous publication changed while jobs queued; reconcile before selection')
        selection = select(plan['run_id'],plan['publication_id'],root,analyses=plan['analyses'])
    write_json(root/'results/provenance'/f"selection-{plan['publication_id']}.json",selection)
    LOG.info('CMI sources and computations verified: %s',validation)
    environment = dict(os.environ,MPLCONFIGDIR=str(root/'logs/matplotlib'),XDG_CACHE_HOME=str(root/'logs/cache'))
    subprocess.run([str(root/'.venv/bin/snakemake'),'verify_computation','paper','--cores','1','--quiet'],
                   cwd=root,env=environment,check=True)
    result = dict(selection,build='passed',validation=validation,pdf_sha256=sha256(root/'paper/main.pdf'))
    write_json(root/'results/provenance'/f"completion-{plan['publication_id']}.json",result)
    LOG.info('CMI TMLE manuscript built: %s',result)
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    complete()
