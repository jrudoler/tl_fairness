"""Compile the current manuscript assets only; no rendering or computation."""
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    env=dict(os.environ,BIBINPUTS=str(ROOT/'paper')+':'+os.environ.get('BIBINPUTS',''))
    subprocess.run(['latexmk','-xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                   cwd=ROOT/'paper',env=env,check=True)
