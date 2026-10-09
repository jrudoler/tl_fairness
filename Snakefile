"""Render the manuscript from verified publication inputs; never launch simulations.

New computation is explicit: use workflow/compute.smk and a fresh run_id.
The configuration registries define artifact lineage and approved input hashes.
"""
import json
import os
import re
from pathlib import Path

PYTHON = os.environ.get("TLFAIR_PYTHON", ".venv/bin/python")
RUN = f"PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {PYTHON}"
REGISTRY = json.loads(Path("config/publication.json").read_text())
ARTIFACTS = json.loads(Path("config/artifacts.json").read_text())["artifacts"]
VALIDATION = "results/provenance/validation.json"
FIGURES = [a["result"] for a in ARTIFACTS if a["result"].endswith(".pdf")]
TABLES = [a["result"] for a in ARTIFACTS if a["result"].endswith(".tex")]
PAPER_ASSETS = [a["paper"] for a in ARTIFACTS]
NUISANCE_TABLES = [f"results/data/nuisance_{name}.tex"
                   for name in ["fitted", "controls", "sensitivity", "population", "findings"]]


def dependencies(*names):
    return sorted({p for a in ARTIFACTS if a["id"] in names for p in a["dependencies"]})


def tex_sources(path="paper/main.tex", seen=None):
    # Follow active inputs only. Graphics and generated tables are declared below.
    seen = set() if seen is None else seen
    if path in seen:
        return seen
    seen.add(path)
    if Path(path).is_file():
        text = re.sub(r"(?<!\\)%[^\n]*", "", Path(path).read_text())
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            source = "paper/" + name.removesuffix(".tex") + ".tex"
            tex_sources(source, seen)
    return seen


rule all:
    input: FIGURES + TABLES


rule validate_publication:
    input:
        data=[r["path"] for r in REGISTRY["inputs"]],
        code=["config/publication.json", "analysis/tools/validate_publication.py",
              "analysis/tools/validate.py", "analysis/nuisance_tables/validate.py",
              "analysis/nuisance_tables/control_audit.py", "experiments/nuisance_training.py",
              "experiments/nuisance_early_stopping.py", "experiments/parity_retraining.py",
              "tlfair/provenance.py", "tlfair/metrics.py", "uv.lock"],
    output: VALIDATION
    log: "logs/validate_publication.log"
    shell: "{RUN} analysis/tools/validate_publication.py --output {output:q} > {log:q} 2>&1"


rule validate:
    input: VALIDATION


rule fig_parity_retraining:
    input:
        verified=VALIDATION,
        deps=dependencies("parity_retraining"),
    output: "results/figures/parity_retraining.pdf"
    log: "logs/fig_parity_retraining.log"
    shell:
        "{RUN} experiments/parity_retraining.py --plot-only "
        "--output-dir data/generated/parity_retraining --figure {output:q} > {log:q} 2>&1"


rule fig_parity_nuisance_training:
    input:
        verified=VALIDATION,
        deps=dependencies("parity_nuisance_training"),
    output: "results/figures/parity_nuisance_training.pdf"
    log: "logs/fig_parity_nuisance_training.log"
    shell: "{RUN} analysis/fig_parity_nuisance_training/run.py --output {output:q} > {log:q} 2>&1"


rule fig_cmi_conditioning_set:
    input:
        verified=VALIDATION,
        deps=dependencies("cmi_conditioning_set"),
    output: "results/figures/cmi_conditioning_set.pdf"
    log: "logs/fig_cmi_conditioning_set.log"
    shell: "{RUN} analysis/fig_cmi_conditioning_set/run.py --output {output:q} > {log:q} 2>&1"


rule fig_cmi_estimation:
    input:
        verified=VALIDATION,
        deps=dependencies("cmi_estimation_error", "cmi_estimation_coverage"),
    output:
        error="results/figures/cmi_estimation_error.pdf",
        coverage="results/figures/cmi_estimation_coverage.pdf",
    log: "logs/fig_cmi_estimation.log"
    shell:
        "{RUN} analysis/fig_cmi_estimation/run.py --error-output {output.error:q} "
        "--coverage-output {output.coverage:q} > {log:q} 2>&1"


rule table_real_data_inference:
    input:
        verified=VALIDATION,
        deps=dependencies("real_data_inference"),
    output:
        csv="results/data/real_data_inference.csv",
        tex="results/data/real_data_inference.tex",
    log: "logs/table_real_data_inference.log"
    shell:
        "{RUN} analysis/table_real_data_inference/run.py --manuscript-only --body-only "
        "--csv-output {output.csv:q} --tex-output {output.tex:q} > {log:q} 2>&1"


rule table_cmi_population_truth:
    input:
        verified=VALIDATION,
        deps=dependencies("cmi_population_truth"),
    output:
        csv="results/data/cmi_population_truth.csv",
        tex="results/data/cmi_population_truth.tex",
    log: "logs/table_cmi_population_truth.log"
    shell:
        "{RUN} analysis/table_cmi_population_truth/run.py --body-only "
        "--csv-output {output.csv:q} --tex-output {output.tex:q} > {log:q} 2>&1"


rule nuisance_tables:
    input:
        verified=VALIDATION,
        deps=dependencies("nuisance_population", "nuisance_fitted", "nuisance_controls", "nuisance_sensitivity"),
    output: NUISANCE_TABLES + ["results/data/nuisance_tables.csv"]
    log: "logs/nuisance_tables.log"
    shell: "{RUN} analysis/nuisance_tables/run.py > {log:q} 2>&1"


rule sync_paper_artifacts:
    input:
        artifacts=FIGURES + TABLES,
        config="config/artifacts.json",
        script="analysis/tools/deliver.py",
        provenance="tlfair/provenance.py",
    output: PAPER_ASSETS + ["results/provenance/delivery.json", "paper/artifact-manifest.json"]
    log: "logs/deliver.log"
    shell: "{RUN} {input.script:q} --config {input.config:q} > {log:q} 2>&1"


rule deliver:
    input: rules.sync_paper_artifacts.output


rule paper:
    input:
        tex=sorted(tex_sources()),
        assets=rules.sync_paper_artifacts.output,
        bib="paper/refs.bib",
        compiler="analysis/tools/compile_paper.py",
    output: "paper/main.pdf"
    log: "logs/paper.log"
    shell: "{RUN} {input.compiler:q} > {log:q} 2>&1"
