"""Execute one explicitly configured new run, separate from publication inputs."""
import argparse
import json
import os
import importlib
import inspect
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tlfair.provenance import git_state, runtime, sha256, write_json


def run(analysis, run_id, config_path, n_jobs=1, reference_dir=None, compare_mem_gb=16):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', run_id):
        raise ValueError('run-id must be a simple directory name')
    if n_jobs < 1 or compare_mem_gb <= 0:
        raise ValueError('Workers and memory budget must be positive')
    specification = json.loads(Path(config_path).read_text())
    settings = specification['analyses'][analysis]
    output = ROOT/'data/runs'/run_id/analysis
    if output.exists():
        raise FileExistsError(f'Use a fresh run ID; refusing to overwrite {output}')
    inputs = {key: ROOT/value for key, value in settings.get('inputs', {}).items()}
    if settings.get('requires_reference'):
        if reference_dir is None:
            raise ValueError('Early stopping requires --reference-dir for a matching reference run')
        reference = Path(reference_dir).resolve()
        for name in ['replicates.csv.gz','population.csv','manifest.json']:
            if not (reference/name).is_file():
                raise FileNotFoundError(reference/name)
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [sys.executable, str(ROOT/settings['script']), '--n-jobs', str(n_jobs)]
    for name, value in settings['parameters'].items():
        command += ['--'+name] + [str(v) for v in (value if isinstance(value, list) else [value])]
    for name, path in {**inputs, **{k: output/v for k,v in settings['outputs'].items()}}.items():
        command += ['--'+name, str(path)]
    if settings.get('requires_reference'):
        command += ['--reference-dir', str(reference)]
    if analysis == 'cmi_estimation':
        command += ['--compare-mem-gb', str(compare_mem_gb)]
    design = {}
    if analysis.startswith('nuisance_'):
        module=importlib.import_module('experiments.'+('nuisance_training' if analysis=='nuisance_reference' else 'nuisance_early_stopping'))
        design={key:getattr(module,key) for key in ['SCENARIOS','SIZES','CONTROL_SIZES','BOOST','HIGH','CASES']}
        design['seed']=inspect.signature(module.one_rep).parameters['seed'].default
        design['evaluation_size']=inspect.signature(module.run_shard).parameters['n_eval'].default
    if analysis.startswith('real_data_'):
        from tlfair.superlearner import SuperLearnerClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier
        seed=settings['parameters']['seed']
        learner=SuperLearnerClassifier(random_state=seed)
        design={'test_fraction':.4,'folds':learner.folds,
                'ensemble':[{'class':type(model).__name__,'parameters':{k:repr(v) for k,v in model.get_params().items()}}
                            for model in learner.models],
                'cmi_outcome':HistGradientBoostingClassifier(random_state=seed).get_params()}
        pinned=json.loads((ROOT/'config/datasets.json').read_text())[analysis.removeprefix('real_data_')]
        if sha256(inputs['input'])!=pinned['canonical_sha256']:
            raise ValueError('Real-data input differs from the pinned dataset')
    source_paths = sorted({ROOT/settings['script'], ROOT/'config/analyses.json', ROOT/'uv.lock',
                           ROOT/'pyproject.toml', *ROOT.joinpath('config').glob('*.json'), *ROOT.joinpath('tlfair').glob('*.py'),
                           *ROOT.joinpath('experiments').glob('*.py'),
                           *ROOT.joinpath('analysis').rglob('*.py')})
    output.mkdir(parents=True)
    sources = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        destination = output/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[str(relative)] = sha256(path)
    shutil.copy2(config_path, output/'resolved_config.json')
    record = dict(analysis=analysis, run_id=run_id, status='running',
                  started=datetime.now(timezone.utc).isoformat(), command=command,
                  settings=settings, design=design, rng_version=specification['rng_version'],
                  resources=dict(workers=n_jobs,compare_mem_gb=compare_mem_gb),
                  runtime=runtime(), git=git_state(), sources=sources,
                  configuration_sha256=sha256(config_path),
                  inputs={str(p):sha256(p) for p in inputs.values()})
    if settings.get('requires_reference'):
        record['reference']={str(reference/p):sha256(reference/p)
                             for p in ['replicates.csv.gz','population.csv','manifest.json']}
    manifest = output/'run_manifest.json'
    write_json(manifest, record)
    try:
        with (output/'execution.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                                    MKL_NUM_THREADS='1',LOKY_MAX_CPU_COUNT=str(n_jobs)))
        for name in settings['required']:
            if not (output/name).is_file():
                raise FileNotFoundError(output/name)
        record['outputs']={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                           if p.is_file() and 'source' not in p.relative_to(output).parts
                           and p != manifest}
        record['status']='complete'
    except Exception as error:
        record['status']='failed'
        record['error']=str(error)
        raise
    finally:
        record['finished']=datetime.now(timezone.utc).isoformat()
        write_json(manifest, record)
    write_json(output/'complete.json',dict(status='complete',manifest_sha256=sha256(manifest)))
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/analyses.json')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--compare-mem-gb', type=float, default=16)
    parser.add_argument('--reference-dir', type=Path)
    args=parser.parse_args()
    print(run(args.analysis,args.run_id,args.config,args.n_jobs,args.reference_dir,args.compare_mem_gb))
