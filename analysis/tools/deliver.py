"""Record checked manuscript artifact copies and their rendering dependencies."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tlfair.provenance import sha256,runtime,git_state,write_json


def deliver(config):
    artifacts=json.loads(Path(config).read_text())['artifacts']
    records=[]
    for item in artifacts:
        source=ROOT/item['result'];destination=ROOT/item['paper']
        records.append({**item, 'sha256': sha256(source),
                        'dependencies': {p:sha256(ROOT/p) for p in item['dependencies']}})
    # Verify every source/dependency before replacing any manuscript asset.
    for item in records:
        source=ROOT/item['result'];destination=ROOT/item['paper']
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        if sha256(destination) != item['sha256']:
            raise ValueError(f'Delivery checksum mismatch: {destination}')
    record=dict(artifacts=records,runtime=runtime(),git=git_state(),
                publication_registry_sha256=sha256(ROOT/'config/publication.json'))
    write_json(ROOT/'results/provenance/delivery.json',record)
    write_json(ROOT/'paper/artifact-manifest.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=ROOT/'config/artifacts.json')
    deliver(parser.parse_args().config)
