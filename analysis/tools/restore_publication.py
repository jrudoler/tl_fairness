"""Restore registered Locust inputs from a verified local bundle, never recompute."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tlfair.provenance import sha256


def restore(bundle):
    registry=json.loads((ROOT/'config/publication.json').read_text())
    recovery=json.loads((ROOT/'config/locust-recovery.json').read_text())
    for item in recovery['files']:
        source=Path(bundle)/item['path']
        if not source.is_file() or sha256(source)!=item['sha256']:
            raise ValueError(f'Missing or changed recovery file: {source}')
    for item in registry['inputs']:
        target=ROOT/item['path']
        if target.exists():
            if sha256(target)!=item['sha256']:
                raise ValueError(f'Quarantine conflicting file before restoring: {target}')
            continue
        prefix='data/imported/locust-2026-10-09/'
        if not item.get('source','').startswith(prefix):
            raise FileNotFoundError(f'Recover the previously delivered nuisance bundle: {target}')
        source=Path(bundle)/item['source'][len(prefix):]
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    print('Publication inputs restored and verified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'data/imported/locust-2026-10-09')
    restore(parser.parse_args().bundle)
