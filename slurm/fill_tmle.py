"""Compute the 3 TMLE inference metrics on full data and merge them into the
existing (inference-only) analyze pickles, then regenerate Table 1.

Reuses run_experiment from the analyze_* entrypoints with importance off, so the
5 one-step metrics already persisted are left untouched and only the TMLE rows
are added. Run via slurm/fill_tmle.sbatch (SuperLearner on 49k rows is too heavy
for the login node).
"""
import importlib.util
import pickle
import sys
from pathlib import Path

ROOT = Path("/home/jrudoler/tl_fairness")
sys.path.insert(0, str(ROOT))

TMLE_METRICS = ["prob_parity_tmle", "prob_opp_tmle", "cmi_tmle"]
GEN = ROOT / "data/generated"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fill(mod, raw_csv, pkl_path):
    res = mod.run_experiment(
        input_path=str(raw_csv),
        importance_samples=0,
        metrics_to_run=TMLE_METRICS,
    )
    with open(pkl_path, "rb") as f:
        existing = pickle.load(f)
    existing["inference"].update(res["inference"])
    existing["timing"].update(res.get("timing", {}))
    with open(pkl_path, "wb") as f:
        pickle.dump(existing, f)
    print(f"merged {list(res['inference'])} into {pkl_path}", flush=True)


def main():
    adult_mod = _load_module("adult_run", ROOT / "analysis/analyze_adult/run.py")
    law_mod = _load_module("law_run", ROOT / "analysis/analyze_law/run.py")
    fill(adult_mod, ROOT / "data/raw/adult.csv", GEN / "adult_results.pkl")
    fill(law_mod, ROOT / "data/raw/law.csv", GEN / "law_results.pkl")

    table1 = _load_module("table1_run", ROOT / "analysis/table1_inference/run.py")
    sys.argv = [
        "table1",
        "--adult-input", str(GEN / "adult_results.pkl"),
        "--law-input", str(GEN / "law_results.pkl"),
        "--csv-output", str(ROOT / "results/data/table1_inference.csv"),
        "--tex-output", str(ROOT / "results/data/table1_inference.tex"),
    ]
    table1.main()


if __name__ == "__main__":
    main()
