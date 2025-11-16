"""Lightweight logging + timing helpers for the GRAVITY Python API."""

from __future__ import annotations

import os
from typing import Any
from contextlib import contextmanager
import time

__all__ = [
    "set_verbose",
    "get_verbose",
    "log_verbose",
    "time_section",
]

_VERBOSITY = int(os.environ.get("GRAVITY_VERBOSE", "2"))
_TIMING_ENABLED = bool(int(os.environ.get("GRAVITY_TIMING", "0")))


def set_verbose(level: int) -> None:
    """Set global verbosity level (higher means more logs)."""

    global _VERBOSITY
    _VERBOSITY = int(level)


def get_verbose() -> int:
    """Return the configured verbosity level."""

    return _VERBOSITY


def log_verbose(message: Any, *, level: int = 1) -> None:
    """Print *message* when the verbosity is at least *level*."""

    if _VERBOSITY >= level:
        print(message)


@contextmanager
def time_section(label: str, *, level: int = 1):
    """Measure wall time for a code section and print when enabled.

    Enable by setting env `GRAVITY_TIMING=1` (or use high `GRAVITY_VERBOSE`).
    Synchronizes CUDA before/after to capture GPU kernels when torch is available.
    """
    # Only print when explicitly enabled via env GRAVITY_TIMING=1
    do_print = _TIMING_ENABLED
    if not do_print:
        yield
        return

    # Try to sync CUDA to include GPU time in measurement
    has_cuda = False
    try:
        import torch  # local import to avoid hard dependency
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.synchronize()
    except Exception:
        has_cuda = False

    t0 = time.perf_counter()
    try:
        yield
    finally:
        if has_cuda:
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[timing] {label}: {dt_ms:.1f} ms")
