"""Checksums and run records shared by recovery, computation, and rendering."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime():
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'joblib', 'snakemake']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def git_state():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain'),
            'paper_commit': subprocess.check_output(['git', '-C', str(ROOT/'paper'),
                                                     'rev-parse', 'HEAD'], text=True).strip()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)
