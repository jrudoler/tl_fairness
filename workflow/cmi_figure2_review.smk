"""Exact Figure 2 protocol, on the frozen CMI diagnostic datasets."""
RUN_DIR = config.get('run_dir', 'data/runs/cmi-figure2-review-20261021')
REPORT_DIR = config.get('report_dir', 'results/review/cmi-figure2-review-20261021')
REFERENCE_DIR = config.get('reference_dir', 'results/review/cmi-learner-review-20261021')

rule all:
    input: REPORT_DIR + '/manifest.json'

rule compare:
    input:
        script='analysis/cmi_learner_review/figure2.py',
        helpers='analysis/cmi_learner_review/run.py',
        protocol='experiments/nuisance_early_stopping.py',
        reference=REFERENCE_DIR + '/replicates.csv.gz',
        reference_manifest=REFERENCE_DIR + '/manifest.json',
    output: REPORT_DIR + '/manifest.json'
    threads: 8
    log: 'logs/cmi_figure2_review.log'
    shell:
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
        '.venv/bin/python {input.script} --output-dir {RUN_DIR} '
        '--report-dir {REPORT_DIR} --reference-dir {REFERENCE_DIR} '
        '--n-jobs {threads} > {log} 2>&1'
