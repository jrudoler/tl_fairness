"""Fast inference-only Table 1 on full data (no permutation importance).

Reuses the EXACT load_data preprocessing from the analyze_* entrypoints so the
numbers are directly comparable to the paper's Table 1, but skips the expensive
perm_importance loop so we get estimates + CIs in minutes. Original 5 metrics
only (the paper has no TMLE rows).
"""

import importlib.util
import pickle
import sys
import time
from pathlib import Path

ROOT = Path("/home/jrudoler/tl_fairness")
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from tlfair.metrics import parity, prob_parity, opportunity, prob_opportunity, cmi  # noqa: E402
from tlfair.superlearner import SuperLearnerClassifier  # noqa: E402

SEED = 123
METRICS = [
    ("parity", parity, "Parity"),
    ("prob_parity", prob_parity, "Prob. Parity"),
    ("opportunity", opportunity, "Eq. Opp."),
    ("prob_opp", prob_opportunity, "Prob. Eq. Opp."),
    ("cmi", cmi, "CMI"),
]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(load_data, csv):
    xtr, xte, ytr, yte, gtr, gte = load_data(csv, SEED)
    out = {}
    for key, fn, _ in METRICS:
        outcome = (HistGradientBoostingClassifier(random_state=SEED) if key == "cmi"
                   else SuperLearnerClassifier(random_state=SEED))
        t = time.perf_counter()
        out[key] = fn(X_train=xtr, X_test=xte, y_train=ytr, y_test=yte, group_train=gtr, group_test=gte,
                      outcome=outcome,
                      propensity=SuperLearnerClassifier(random_state=SEED))
        est, ci = out[key]
        print(f"{Path(csv).name} {key}: {est:.4f} ({ci[0]:.4f}, {ci[1]:.4f}) "
              f"[{time.perf_counter() - t:.0f}s]", flush=True)
    return out


def main():
    adult_mod = _load_module("adult_run", ROOT / "analysis/analyze_adult/run.py")
    law_mod = _load_module("law_run", ROOT / "analysis/analyze_law/run.py")

    res = {
        "adult": run(adult_mod.load_data, str(ROOT / "data/raw/adult.csv")),
        "law": run(law_mod.load_data, str(ROOT / "data/raw/law.csv")),
    }
    with open("/tmp/quick_table1.pkl", "wb") as f:
        pickle.dump(res, f)

    print("\nMetric, Adult, Law", flush=True)
    for key, _, label in METRICS:
        a, l = res["adult"][key], res["law"][key]
        print(f"{label}, {a[0]:.2f} ({a[1][0]:.2f}, {a[1][1]:.2f}), "
              f"{l[0]:.2f} ({l[1][0]:.2f}, {l[1][1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
