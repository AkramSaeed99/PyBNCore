"""Owns the live `PyBNCoreWrapper` and serializes cross-thread access."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from PySide6.QtCore import QMutex

# Deliberate late binding — `pybncore` is only importable on the service side.
# We keep the type annotation as a string to avoid pulling it into GUI-adjacent
# code paths that don't need it.
try:
    from pybncore import PyBNCoreWrapper  # noqa: F401
except ImportError:  # pragma: no cover — runtime environment without the engine
    PyBNCoreWrapper = object  # type: ignore[assignment,misc]


class ModelSession:
    """Single owner of the compiled model and a mutex guarding it.

    All services enter `with session.locked() as wrapper:` before touching the
    engine. This keeps multiple workers from stepping on each other during
    compile / query.
    """

    def __init__(self) -> None:
        self._wrapper: Optional["PyBNCoreWrapper"] = None
        self._mutex = QMutex()
        self._compiled = False
        self._source_path: Optional[str] = None

    @contextmanager
    def locked(self) -> Iterator[Optional["PyBNCoreWrapper"]]:
        self._mutex.lock()
        try:
            yield self._wrapper
        finally:
            self._mutex.unlock()

    def set_wrapper(self, wrapper: "PyBNCoreWrapper", source_path: Optional[str] = None) -> None:
        self._mutex.lock()
        try:
            self._wrapper = wrapper
            self._source_path = source_path
            self._compiled = False
        finally:
            self._mutex.unlock()

    def set_source_path(self, source_path: Optional[str]) -> None:
        self._mutex.lock()
        try:
            self._source_path = source_path
        finally:
            self._mutex.unlock()

    def clear(self) -> None:
        self._mutex.lock()
        try:
            self._wrapper = None
            self._source_path = None
            self._compiled = False
        finally:
            self._mutex.unlock()

    @property
    def has_model(self) -> bool:
        return self._wrapper is not None

    @property
    def is_compiled(self) -> bool:
        return self._compiled

    @property
    def source_path(self) -> Optional[str]:
        return self._source_path

    def mark_compiled(self) -> None:
        self._compiled = True

    def invalidate_compile(self) -> None:
        self._compiled = False
