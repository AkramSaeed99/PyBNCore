"""QUndoCommand subclasses for edge mutations."""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtGui import QUndoCommand

from pybncore_gui.services.authoring_service import AuthoringService


class _BaseEdgeCommand(QUndoCommand):
    def __init__(
        self,
        text: str,
        service: AuthoringService,
        on_structure_changed: Callable[[], None],
    ) -> None:
        super().__init__(text)
        self._service = service
        self._notify = on_structure_changed


class AddEdgeCommand(_BaseEdgeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: Callable[[], None],
        parent: str,
        child: str,
    ) -> None:
        super().__init__(f"Add edge {parent} → {child}", service, on_structure_changed)
        self._parent = parent
        self._child = child
        self._prior_child_cpt: np.ndarray | None = None

    def redo(self) -> None:  # type: ignore[override]
        self._prior_child_cpt = self._service.add_edge(self._parent, self._child)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        self._service.remove_edge(self._parent, self._child)
        if self._prior_child_cpt is not None and self._prior_child_cpt.size:
            self._service.set_flat_cpt(self._child, self._prior_child_cpt)
        self._notify()


class RemoveEdgeCommand(_BaseEdgeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: Callable[[], None],
        parent: str,
        child: str,
    ) -> None:
        super().__init__(f"Remove edge {parent} → {child}", service, on_structure_changed)
        self._parent = parent
        self._child = child
        self._prior_child_cpt: np.ndarray | None = None

    def redo(self) -> None:  # type: ignore[override]
        self._prior_child_cpt = self._service.remove_edge(self._parent, self._child)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        self._service.add_edge(self._parent, self._child)
        if self._prior_child_cpt is not None and self._prior_child_cpt.size:
            self._service.set_flat_cpt(self._child, self._prior_child_cpt)
        self._notify()
