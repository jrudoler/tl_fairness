"""Compile current manuscript assets; tolerate compute nodes without Perl modules."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def compile_manuscript(root: Path = ROOT) -> None:
    env = dict(os.environ, BIBINPUTS=str(root/'paper')+':'+os.environ.get('BIBINPUTS',''))
    for key, default in [('XDG_CACHE_HOME','logs/cache'),('MPLCONFIGDIR','logs/matplotlib')]:
        cache = Path(env.get(key,default)).expanduser()
        if not cache.is_absolute():
            cache = root/cache
        cache.mkdir(parents=True,exist_ok=True)
        env[key] = str(cache)
    probe = subprocess.run(['perl','-MTime::HiRes','-e','1'],env=env,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    if probe.returncode == 0 and shutil.which('latexmk'):
        subprocess.run(['latexmk','-g','-xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                       cwd=root/'paper',env=env,check=True)
        return
    LOG.info('Using direct XeLaTeX/BibTeX passes because latexmk or Perl Time::HiRes is unavailable')
    command = ['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex']
    subprocess.run(command,cwd=root/'paper',env=env,check=True)
    subprocess.run(['bibtex','main'],cwd=root/'paper',env=env,check=True)
    for _ in range(5):
        subprocess.run(command,cwd=root/'paper',env=env,check=True)
        log = (root/'paper/main.log').read_text(errors='replace')
        if not any(message in log for message in ['Rerun to get cross-references right',
                                                   'Label(s) may have changed','There were undefined references',
                                                   'There were undefined citations']):
            return
    raise RuntimeError('Manuscript references did not settle after five XeLaTeX passes')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    compile_manuscript()
