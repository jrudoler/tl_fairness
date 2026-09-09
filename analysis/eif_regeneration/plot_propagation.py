"""Redraw the archived propagation experiment with roundoff-safe error bars."""
import argparse
import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/out/eif_regeneration'))
    args = parser.parse_args()
    out = ROOT / args.output_dir
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    loader = importlib.machinery.SourcelessFileLoader('archived_propagation',
        str(out/'before/exp6_propagation.cpython-313.pyc'))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    capacity = pd.read_csv(out/'exp6_propagation.csv')
    fluke = pd.read_csv(out/'exp6_propagation_fluke.csv')
    truths = capacity.groupby('violation_scale').psi.first().to_dict()
    original = Axes.errorbar

    def rounded_errorbar(self: Axes, *args: object, **kwargs: object) -> object:
        if 'yerr' in kwargs:
            error = np.asarray(kwargs['yerr'])
            if np.any(error < 0):
                minimum = float(error.min())
                if minimum < -1e-12:
                    raise ValueError(f'Negative error exceeds roundoff tolerance: {minimum}')
                logging.info('Clipped roundoff-only negative error bar: %.3g', minimum)
                kwargs['yerr'] = np.maximum(error, 0)
        return original(self, *args, **kwargs)

    with patch.object(Axes, 'errorbar', rounded_errorbar):
        module.plot(fluke, capacity, truths, out/'exp6_propagation.png', tau_ref=0.02)
    fig = plt.gcf()
    titles = ['Model disparity vs. capacity', 'Disparity vs. prediction error',
              'Fair-looking models among detections']
    for ax, title in zip(fig.axes, titles):
        ax.set_title(title, fontsize=8)
    fig.savefig(out/'exp6_propagation.png')
    plt.close('all')


if __name__ == '__main__':
    main()
