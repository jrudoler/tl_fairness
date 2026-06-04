from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_repo_root = Path(__file__).resolve().parent.parent
_repo_root_str = str(_repo_root)
if _repo_root_str not in __path__:
    __path__.append(_repo_root_str)

from . import experiments, tlfair  # noqa: E402

__all__ = ["experiments", "tlfair"]
