"""Shared fixtures for GUI service tests.

A single `QCoreApplication` is spun up once per session so `QMutex` /
`QThread` primitives used inside the services can operate without a
full GUI.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def qt_app():
    try:
        from PySide6.QtCore import QCoreApplication
    except ImportError:  # pragma: no cover — GUI extras not installed
        pytest.skip("PySide6 not available", allow_module_level=True)
        return None
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)
    yield app


@pytest.fixture
def asia_mini_path() -> pathlib.Path:
    path = FIXTURE_DIR / "asia_mini.xdsl"
    assert path.is_file(), f"Missing fixture: {path}"
    return path
