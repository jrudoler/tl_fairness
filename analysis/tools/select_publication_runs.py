"""Select certified run outputs, retaining previous inputs outside Git."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import re
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.tools.audit_publication_runs import audit
from analysis.tools.validate_publication import verify_inputs
from tlfair.provenance import sha256, write_json

LOG = logging.getLogger(__name__)


def select(run_id: str, publication_id: str, root: Path = ROOT) -> dict[str, Any]:
    for value in [run_id, publication_id]:
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', value):
            raise ValueError('Run and publication IDs must be simple directory names')
    publication_path = root/'config/publication.json'
    selection_path = root/'config/publication-runs.json'
    previous = json.loads(publication_path.read_text())
    verify_inputs(previous, root)
    selections = json.loads(selection_path.read_text())
    proposed = copy.deepcopy(selections)
    for analysis, item in proposed['analyses'].items():
        item['run'] = f'data/runs/{run_id}/{analysis}'
        item.pop('manifest', None)
        item.pop('complete', None)
        item['note'] = 'Selected completed computation; source snapshot, current computation code, inputs, runtime and outputs verified.'
    preflight = audit(root, proposed['analyses'], verify_publication_inputs=False)
    if preflight['status'] != 'verified':
        raise ValueError('Run selection failed preflight: '+json.dumps(preflight))
    archive = root/'data/imported'/f'publication-before-{publication_id}'
    if archive.exists():
        raise FileExistsError(f'Previous-input archive already exists: {archive}')
    archive.mkdir(parents=True)
    for path in [publication_path, selection_path]:
        target = archive/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,target)
    moved, created = [], []
    entries = [item for item in previous['inputs'] if item['path'].startswith('data/raw/')]
    for item in entries:
        item['source_kind'] = 'pinned-dataset'
        item['source'] = 'config/datasets.json'
    try:
        # Retain all former generated inputs, including retired delivery logs
        # and summaries, before copying new immutable run outputs into place.
        for item in previous['inputs']:
            path = root/item['path']
            if item['path'].startswith('data/generated/'):
                target = archive/item['path']
                target.parent.mkdir(parents=True,exist_ok=True)
                path.replace(target)
                moved.append((path,target))
        for analysis, item in proposed['analyses'].items():
            directory = root/item['run']
            record = json.loads((directory/'run_manifest.json').read_text())
            for field, name in [('manifest','run_manifest.json'),('complete','complete.json')]:
                destination = root/'config/publication-provenance'/analysis/name
                if destination.exists():
                    backup = archive/destination.relative_to(root)
                    backup.parent.mkdir(parents=True,exist_ok=True)
                    destination.replace(backup)
                    moved.append((destination,backup))
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(directory/name,destination)
                created.append(destination)
                item[field] = str(destination.relative_to(root))
                entries.append({'path':item[field],'sha256':sha256(destination),
                                'source':str((directory/name).relative_to(root)),
                                'source_kind':'completed-run-record'})
            for published, original in item['inputs'].items():
                source, destination = directory/original, root/published
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,destination)
                created.append(destination)
                expected = record['outputs'][original]
                if sha256(destination) != expected:
                    raise ValueError(f'Copy differs from certified output: {published}')
                entries.append({'path':published,'sha256':expected,
                                'source':str(source.relative_to(root)),
                                'source_kind':'completed-run-output','run_manifest':item['manifest'],
                                'run_manifest_sha256':sha256(directory/'run_manifest.json')})
            # These native run records establish the CMI summary provenance;
            # retain them in place rather than duplicating them canonically.
            evidence = {'cmi_estimation':['coverage_replicates.csv.gz','comparison_replicates.csv.gz'],
                        'cmi_conditioning_set':['replicates.csv.gz']}.get(analysis,[])
            for name in evidence:
                path = str((directory/name).relative_to(root))
                entries.append({'path':path,'sha256':record['outputs'][name],
                                'source_kind':'completed-run-replicates','run_manifest':item['manifest']})
        publication = {'schema_version':2,'publication_id':publication_id,
                       'inputs':entries,'limitations':[
                           'CMI Wald inference is not calibrated at conditional independence; provenance verification does not establish statistical validity.',
                           'Exact numerical reproduction requires the recorded dependency versions and source snapshot.']}
        write_json(selection_path,proposed)
        write_json(publication_path,publication)
        verification = audit(root)
        if verification['status'] != 'verified':
            raise ValueError('Selected publication failed verification: '+json.dumps(verification))
        verify_inputs(publication,root)
        return {'publication_id':publication_id,'run_id':run_id,'status':'verified',
                'previous_inputs':str(archive.relative_to(root)),'input_count':len(entries)}
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        for path,backup in reversed(moved):
            path.parent.mkdir(parents=True,exist_ok=True)
            backup.replace(path)
        for path in [publication_path,selection_path]:
            shutil.copy2(archive/path.relative_to(root),path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--publication-id',required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    result = select(args.run_id,args.publication_id)
    write_json(ROOT/'results/provenance'/f'selection-{args.publication_id}.json',result)
    LOG.info('Selected publication: %s',result)


if __name__ == '__main__':
    main()
