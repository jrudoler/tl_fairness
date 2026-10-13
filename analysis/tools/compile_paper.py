"""Compile the current manuscript assets only; no rendering or computation."""
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    env=dict(os.environ,BIBINPUTS=str(ROOT/'paper')+':'+os.environ.get('BIBINPUTS',''))
    for key, default in [('XDG_CACHE_HOME','logs/cache'),('MPLCONFIGDIR','logs/matplotlib')]:
        cache = Path(env.get(key, default)).expanduser()
        if not cache.is_absolute():
            cache = ROOT/cache
        cache.mkdir(parents=True,exist_ok=True)
        env[key] = str(cache)
    subprocess.run(['latexmk','-g','-xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                   cwd=ROOT/'paper',env=env,check=True)
