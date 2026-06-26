"""Project-root resolution for the llava_viq package.

``PROJECT_ROOT`` is the directory that should be on ``sys.path`` so that
``import llava_viq`` works -- i.e. the parent of this package directory.

Resolution order:
1. ``$VIQ_ROOT`` if set (lets the launch script pin it explicitly).
2. Otherwise inferred as the parent of the ``llava_viq`` package dir.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["PROJECT_ROOT", "PACKAGE_ROOT", "ensure_on_sys_path", "VIQ_ROOT_ENV"]

VIQ_ROOT_ENV = "VIQ_ROOT"

# this file lives at <PROJECT_ROOT>/llava_viq/_paths.py
PACKAGE_ROOT: Path = Path(__file__).resolve().parent          # .../llava_viq
_inferred_project_root: Path = PACKAGE_ROOT.parent            # .../viq_train

PROJECT_ROOT: Path = Path(
    os.environ.get(VIQ_ROOT_ENV, str(_inferred_project_root))
).resolve()


def ensure_on_sys_path(*extra: "str | os.PathLike") -> None:
    """Prepend ``PROJECT_ROOT`` (and any ``extra`` dirs) to ``sys.path``. Idempotent."""
    for p in [str(PROJECT_ROOT), *(str(Path(e).resolve()) for e in extra)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# Side effect on import: make the package importable regardless of cwd.
ensure_on_sys_path()
