"""nix-llama: turnkey llama.cpp serving with hardware-aware defaults."""

from __future__ import annotations

import os
import pathlib

__all__ = ["__version__"]


def _installed_version() -> str | None:
    """The version recorded by the installer, if there is one."""
    try:
        from importlib.metadata import version

        return version("nix-llama")
    except Exception:
        # A bare source checkout has no installed metadata. Not an error.
        return None


def _read_version() -> str:
    """Resolve the package version without importing anything heavy.

    The Nix build stamps ``NIX_LLAMA_VERSION`` so the binary always reports the
    flake's version, installed copies fall back to their recorded metadata, and
    a source checkout reads the ``VERSION`` file at the repository root.
    """
    env = os.environ.get("NIX_LLAMA_VERSION")
    if env:
        return env

    installed = _installed_version()
    if installed:
        return installed

    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "VERSION"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return "0.0.0+unknown"


__version__ = _read_version()
