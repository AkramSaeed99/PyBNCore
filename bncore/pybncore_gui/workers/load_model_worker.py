"""Load an XDSL or BIF file on a background thread."""
from __future__ import annotations

from dataclasses import dataclass

from pybncore_gui.services.io_service import IOService
from pybncore_gui.workers.base_worker import BaseWorker


@dataclass(frozen=True, slots=True)
class LoadModelResult:
    path: str
    kind: str   # "xdsl" | "bif"


class LoadModelWorker(BaseWorker):
    """Perform `open_xdsl` / `import_bif` off the UI thread."""

    def __init__(self, io_service: IOService, path: str, kind: str = "xdsl") -> None:
        super().__init__()
        self._io = io_service
        self._path = path
        self._kind = kind

    def _execute(self) -> object:
        self.progress.emit(10, f"Reading {self._path}…")
        if self._kind == "bif":
            self._io.import_bif(self._path)
        else:
            self._io.open_xdsl(self._path)
        self.progress.emit(100, "Model loaded")
        return LoadModelResult(path=self._path, kind=self._kind)
