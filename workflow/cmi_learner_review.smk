"""Explicit diagnostic workflow, independent of manuscript input selection."""
RUN_DIR = config.get('run_dir', 'data/runs/cmi-learner-review-20261021')
REPORT_DIR = config.get('report_dir', 'results/review/cmi-learner-review-20261021')
PYTHON = '.venv/bin/python'
RUN = f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {PYTHON}'
RECEIPTS = expand(RUN_DIR + '/kappa_{kappa}/complete_{replicate}.json',
                  kappa=['0', '2.5', '4'], replicate=[f'{r:03d}' for r in range(100)])

rule all:
    input: f'{REPORT_DIR}/manifest.json'

rule prepare:
    output: f'{RUN_DIR}/manifest.json'
    shell: '{RUN} analysis/cmi_learner_review/run.py prepare --output-dir {RUN_DIR}'

rule compare:
    input: rules.prepare.output
    output: RECEIPTS
    threads: 4
    log: 'logs/cmi_learner_review.log'
    shell:
        'for kappa in 0 2.5 4; do {RUN} analysis/cmi_learner_review/run.py run '
        '--output-dir {RUN_DIR} --kappa "$kappa" --reps 100 --n-jobs {threads} '
        '>> {log} 2>&1; done'

rule collect:
    input: rules.compare.output
    output: f'{REPORT_DIR}/manifest.json'
    log: 'logs/cmi_learner_review_collect.log'
    shell:
        'MPLCONFIGDIR=logs/matplotlib XDG_CACHE_HOME=logs/cache {RUN} '
        'analysis/cmi_learner_review/run.py collect --output-dir {RUN_DIR} '
        '--report-dir {REPORT_DIR} > {log} 2>&1'
