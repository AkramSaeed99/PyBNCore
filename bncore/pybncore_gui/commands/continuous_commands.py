"""QUndoCommand subclasses for continuous-node authoring."""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QUndoCommand

from pybncore_gui.domain.continuous import ContinuousNodeSpec
from pybncore_gui.services.authoring_service import AuthoringService


class AddContinuousNodeCommand(QUndoCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: Callable[[], None],
        spec: ContinuousNodeSpec,
    ) -> None:
        super().__init__(f"Add continuous node '{spec.name}'")
        self._service = service
        self._notify = on_structure_changed
        self._spec = spec

    def redo(self) -> None:  # type: ignore[override]
        self._service.add_continuous_node(self._spec)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        try:
            self._service.remove_node(self._spec.name)
        finally:
            self._notify()
