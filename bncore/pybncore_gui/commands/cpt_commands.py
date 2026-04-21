"""QUndoCommand for CPT edits."""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtGui import QUndoCommand

from pybncore_gui.services.authoring_service import AuthoringService


class SetCPTCommand(QUndoCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: Callable[[], None],
        node: str,
        shaped_2d: np.ndarray,
    ) -> None:
        super().__init__(f"Edit CPT for '{node}'")
        self._service = service
        self._notify = on_structure_changed
        self._node = node
        self._new = np.asarray(shaped_2d, dtype=np.float64).copy()
        self._old_flat: np.ndarray | None = None

    def redo(self) -> None:  # type: ignore[override]
        self._old_flat = self._service.set_cpt(self._node, self._new)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        if self._old_flat is not None and self._old_flat.size:
            self._service.set_flat_cpt(self._node, self._old_flat)
            self._notify()
