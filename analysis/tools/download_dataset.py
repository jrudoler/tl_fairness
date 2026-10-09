"""Download pinned, checksummed original dataset bytes without reserialization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tlfair.provenance import sha256,write_json


def download(dataset,output=None):
    specification=json.loads((ROOT/'config/datasets.json').read_text())[dataset]
    output=Path(output) if output else ROOT/'data/raw'/f'{dataset}.csv'
    if output.exists():
        if sha256(output)!=specification['canonical_sha256']:
            raise ValueError(f'Refusing to replace conflicting dataset: {output}')
    else:
        raw=urlopen(specification['url'],timeout=60).read()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=specification['download_sha256'] or digest!=specification['canonical_sha256']:
            raise ValueError('Pinned source does not match the recorded original dataset bytes')
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix(output.suffix+'.tmp');temporary.write_bytes(raw);temporary.replace(output)
    write_json(output.with_suffix('.provenance.json'),specification)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['adult','law']);parser.add_argument('--output',type=Path)
    args=parser.parse_args();print(download(args.dataset,args.output))
