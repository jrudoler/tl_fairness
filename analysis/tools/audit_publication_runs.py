"""Link manuscript inputs to completed computations using the current code."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tlfair.provenance import runtime, sha256, write_json

LOG = logging.getLogger(__name__)


def dependencies(root: Path = ROOT) -> list[str]:
    """Files that invalidate a prior audit when their contents change."""
    paths = {'config/analyses.json', 'config/publication-runs.json', 'config/artifacts.json',
             'analysis/tools/audit_publication_runs.py', 'tlfair/provenance.py'}
    specifications = json.loads((root/'config/analyses.json').read_text())['analyses']
    registry = json.loads((root/'config/publication-runs.json').read_text())['analyses']
    for analysis, selection in registry.items():
        paths.update(specifications[analysis]['code'])
        paths.update(selection['inputs'])
        paths.update(specifications[analysis].get('inputs', {}).values())
        paths.update(selection[field] for field in ['manifest','complete'] if field in selection)
        if selection['run'] is not None:
            directory = root/selection['run']
            paths.update(str((directory/name).relative_to(root)) for name in ['run_manifest.json', 'complete.json'])
            manifest = directory/'run_manifest.json'
            if manifest.is_file():
                record = json.loads(manifest.read_text())
                paths.update(str((directory/'source'/p).relative_to(root)) for p in record.get('sources', {}))
                # Final replicate bundles and selected inputs contain the
                # scientific evidence. Individual shard caches are retained
                # in the run manifest without expanding the publication DAG
                # into tens of thousands of intermediate files.
                outputs = set(specifications[analysis]['required']) | set(selection['inputs'].values())
                paths.update(str((directory/p).relative_to(root)) for p in outputs)
    return sorted(paths)


def audit(root: Path = ROOT, registry: dict[str, Any] | None = None,
          verify_publication_inputs: bool = True) -> dict[str, Any]:
    specifications = json.loads((root/'config/analyses.json').read_text())['analyses']
    registry = registry if registry is not None else json.loads((root/'config/publication-runs.json').read_text())['analyses']
    current_runtime = runtime()
    reports = {}
    global_issues = []
    artifacts = json.loads((root/'config/artifacts.json').read_text())['artifacts']
    claimed = {path for selection in registry.values() for path in selection['inputs']}
    required = {p for artifact in artifacts for p in artifact['dependencies'] if p.startswith('data/generated/')}
    for path in sorted(required - claimed):
        global_issues.append(f'Manuscript computation dependency has no run mapping: {path}')
    for analysis, selection in registry.items():
        issues = []
        report = {'run': selection['run'], 'note': selection.get('note', '')}
        if selection['run'] is None:
            issues.append('No completed computation is selected for these publication inputs.')
        else:
            try:
                directory = root/selection['run']
                manifest = directory/'run_manifest.json'
                record = json.loads(manifest.read_text())
                complete = json.loads((directory/'complete.json').read_text())
                for field, original in [('manifest', manifest), ('complete', directory/'complete.json')]:
                    if field in selection and sha256(root/selection[field]) != sha256(original):
                        issues.append(f'Tracked {field} differs from the completed run record.')
                report['git'] = record.get('git')
                report['code'] = record.get('code', {})
                report['runtime'] = record.get('runtime', {})
                if complete.get('status') != 'complete' or complete.get('manifest_sha256') != sha256(manifest):
                    issues.append('Completion record does not certify this run manifest.')
                if record.get('status') != 'complete' or record.get('analysis') != analysis:
                    issues.append('Run is incomplete or belongs to a different analysis.')
                if record.get('execution') != 'source_snapshot':
                    issues.append('Run did not certify execution of its saved source snapshot.')
                if record.get('settings') != specifications[analysis]:
                    issues.append('Recorded scientific settings differ from the current configuration.')
                if record.get('rng_version') != json.loads((root/'config/analyses.json').read_text())['rng_version']:
                    issues.append('Recorded RNG version differs from the current configuration.')
                for path in specifications[analysis]['code']:
                    expected = record.get('code', {}).get(path)
                    if expected is None or sha256(root/path) != expected:
                        issues.append(f'Current computation dependency differs: {path}')
                    if expected is None or record.get('sources', {}).get(path) != expected:
                        issues.append(f'Computation dependency is not certified in the executed snapshot: {path}')
                for path, expected in record.get('sources', {}).items():
                    if sha256(directory/'source'/path) != expected:
                        issues.append(f'Saved computation source differs: {path}')
                for path, expected in record.get('input_files', {}).items():
                    if sha256(root/path) != expected:
                        issues.append(f'Current raw input differs: {path}')
                for name, path in specifications[analysis].get('inputs', {}).items():
                    if path not in record.get('input_files', {}):
                        issues.append(f'Run has no input checksum for {name}: {path}')
                outputs = set(specifications[analysis]['required']) | set(selection['inputs'].values())
                for path in sorted(outputs):
                    expected = record.get('outputs', {}).get(path)
                    if expected is None:
                        issues.append(f'Required output has no checksum: {path}')
                        continue
                    if sha256(directory/path) != expected:
                        issues.append(f'Recorded run output differs: {path}')
                for path in specifications[analysis]['required']:
                    if path not in record.get('outputs', {}):
                        issues.append(f'Required output has no checksum: {path}')
                if specifications[analysis].get('requires_reference'):
                    reference = registry['nuisance_reference']['run']
                    if reference is None:
                        issues.append('No current-code nuisance reference is selected.')
                    else:
                        # The early-stopping control audit must use the same
                        # reference that is selected for the paper.
                        for path in ['replicates.csv.gz', 'population.csv', 'manifest.json']:
                            expected = sha256(root/reference/path)
                            matches = [v for p,v in record.get('reference', {}).items() if p.endswith('/'+path)]
                            if matches != [expected]:
                                issues.append(f'Run used a different nuisance reference: {path}')
                if verify_publication_inputs:
                    for published, original in selection['inputs'].items():
                        expected = record.get('outputs', {}).get(original)
                        if expected is None or sha256(root/published) != expected:
                            issues.append(f'Publication input is not the recorded run output: {published}')
                for package, version in record.get('runtime', {}).items():
                    if package != 'platform' and current_runtime.get(package) != version:
                        issues.append(f'Current runtime differs for {package}: {version}')
            except (OSError, ValueError, KeyError) as error:
                issues.append(f'Cannot verify selected run: {error}')
        report['issues'] = issues
        report['status'] = 'verified' if not issues else 'unverified'
        reports[analysis] = report
    return {'status': 'verified' if not global_issues and all(r['status'] == 'verified' for r in reports.values()) else 'unverified',
            'issues': global_issues, 'analyses': reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/provenance/computation-audit.json')
    parser.add_argument('--strict', action='store_true', help='Fail unless every selected computation verifies.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = audit()
    write_json(args.output, result)
    for name, record in result['analyses'].items():
        LOG.info('%s: %s — %s', name, record['status'], '; '.join(record['issues']) or record['run'])
    for issue in result['issues']:
        LOG.error('%s', issue)
    if args.strict and result['status'] != 'verified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
