"""Explicit new computation; never included by the publication Snakefile."""
import json
import os
import shlex
from pathlib import Path

ANALYSIS = config["analysis"]
RUN_ID = config["run_id"]
SPEC = json.loads(Path("config/analyses.json").read_text())["analyses"][ANALYSIS]
PYTHON = os.environ.get("TLFAIR_PYTHON", ".venv/bin/python")
REFERENCE = config.get("reference_dir")
if SPEC.get("requires_reference") and REFERENCE is None:
    raise ValueError("Set reference_dir to a matching, completed nuisance reference run")

rule compute:
    input:
        source=["config/analyses.json", "analysis/tools/run_analysis.py", "pyproject.toml", "uv.lock"]
            + sorted(str(p) for folder in ["tlfair", "experiments", "analysis"]
                     for p in Path(folder).rglob("*.py") if "out" not in p.parts),
        data=list(SPEC.get("inputs", {}).values()),
        reference=[] if REFERENCE is None else [f"{REFERENCE}/{p}" for p in ["replicates.csv.gz", "population.csv", "manifest.json"]],
    output: f"data/runs/{RUN_ID}/{ANALYSIS}/complete.json"
    threads: int(config.get("njobs", 1))
    params:
        reference="" if REFERENCE is None else f"--reference-dir {shlex.quote(str(REFERENCE))}",
        memory=float(config.get("compare_mem_gb", 16)),
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {PYTHON} "
        "analysis/tools/run_analysis.py --analysis {ANALYSIS:q} --run-id {RUN_ID:q} "
        "--n-jobs {threads} --compare-mem-gb {params.memory} {params.reference}"
