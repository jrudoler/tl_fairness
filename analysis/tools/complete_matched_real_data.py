"""Select and build the approved matched comparison after both Slurm fits succeed."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.tools.select_publication_runs import select
from analysis.tools.validate_publication import validate_matched_predictions
from tlfair.provenance import sha256, write_json

LOG = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    plan = json.loads((ROOT/'config/matched-real-data-publication.json').read_text())
    manuscript = ROOT/plan['manuscript_source']
    draft = ROOT/plan['manuscript_draft']
    if sha256(manuscript) != plan['manuscript_before_sha256'] or sha256(draft) != plan['manuscript_draft_sha256']:
        raise ValueError('Manuscript or approved draft changed while jobs were queued; reconcile before selecting.')
    additional = {f'model_fairness_{dataset}': {
        'inputs': {f'data/generated/{dataset}_model_comparison.pkl': 'comparison.pkl'}}
        for dataset in ['adult', 'law']}
    result = select(plan['run_id'], plan['publication_id'], analyses=list(additional), additional=additional)
    LOG.info('Selected verified comparisons: %s', result)
    validate_matched_predictions()
    artifacts_path = ROOT/'config/artifacts.json'
    artifacts = json.loads(artifacts_path.read_text())
    for artifact in artifacts['artifacts']:
        if artifact['id'] == 'real_data_inference':
            for dataset in ['adult', 'law']:
                path = f'data/generated/{dataset}_model_comparison.pkl'
                if path not in artifact['dependencies']:
                    artifact['dependencies'].append(path)
    write_json(artifacts_path, artifacts)
    shutil.copy2(draft, manuscript)
    environment = dict(os.environ, MPLCONFIGDIR=str(ROOT/'logs/matplotlib'), XDG_CACHE_HOME=str(ROOT/'logs/cache'))
    subprocess.run([str(ROOT/'.venv/bin/snakemake'), 'verify_computation', 'paper', '--cores', '1', '--quiet'],
                   cwd=ROOT, env=environment, check=True)
    write_json(ROOT/'results/provenance'/f"selection-{plan['publication_id']}.json", result)
    LOG.info('Matched comparison selected, validated, delivered and manuscript compiled.')


if __name__ == '__main__':
    main()
