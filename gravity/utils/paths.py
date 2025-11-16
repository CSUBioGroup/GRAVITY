"""Path-resolution helpers for GRAVITY."""

from __future__ import annotations

from pathlib import Path
from typing import Union

__all__ = ["resolve_path"]


def resolve_path(path: Union[str, Path], *, must_exist: bool = True) -> Path:
    """Resolve *path* to an absolute path.

    The function first expands user symbols, then checks the following locations:
    1. Absolute paths are returned as-is (after resolving symlinks).
    2. Relative to the current working directory.
    3. Relative to the GRAVITY project root (two levels above this file).

    Parameters
    ----------
    path:
        Path-like input provided by the user.
    must_exist:
        When ``True`` (default) raise ``FileNotFoundError`` if the resolved path
        does not exist.
    """

    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        resolved = candidate.resolve()
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Resource not found: {path}")
        return resolved

    search_roots = []
    search_roots.append(Path.cwd())
    pkg_root = Path(__file__).resolve().parents[2]
    search_roots.append(pkg_root)
    for ancestor in pkg_root.parents:
        search_roots.append(ancestor)

    for root in search_roots:
        candidate_path = (root / candidate).resolve()
        if candidate_path.exists():
            return candidate_path

    resolved = (pkg_root / candidate).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Resource not found: {path}")

    return resolved
