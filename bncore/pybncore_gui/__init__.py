"""PyBNCore desktop GUI (Phase 1)."""
from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    from pybncore_gui.app import main as _main
    return _main()
