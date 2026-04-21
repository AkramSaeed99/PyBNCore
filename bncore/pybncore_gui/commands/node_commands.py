"""QUndoCommand subclasses for node-scoped mutations."""
from __future__ import annotations

from typing import Callable, Sequence

from PySide6.QtGui import QUndoCommand

from pybncore_gui.services.authoring_service import AuthoringService, NodeSnapshot

MOVE_COMMAND_ID = 0x1001

StructureChangedCallback = Callable[[], None]


class _BaseNodeCommand(QUndoCommand):
    def __init__(
        self,
        text: str,
        service: AuthoringService,
        on_structure_changed: StructureChangedCallback,
    ) -> None:
        super().__init__(text)
        self._service = service
        self._notify = on_structure_changed


class AddNodeCommand(_BaseNodeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: StructureChangedCallback,
        name: str,
        states: Sequence[str],
        parents: Sequence[str] = (),
    ) -> None:
        super().__init__(f"Add node '{name}'", service, on_structure_changed)
        self._name = name
        self._states = tuple(states)
        self._parents = tuple(parents)

    def redo(self) -> None:  # type: ignore[override]
        self._service.add_discrete_node(self._name, self._states, self._parents)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        self._service.remove_node(self._name)
        self._notify()


class RemoveNodeCommand(_BaseNodeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: StructureChangedCallback,
        name: str,
    ) -> None:
        super().__init__(f"Remove node '{name}'", service, on_structure_changed)
        self._name = name
        self._snapshot: NodeSnapshot | None = None

    def redo(self) -> None:  # type: ignore[override]
        self._snapshot = self._service.remove_node(self._name)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        if self._snapshot is None:
            return
        snap = self._snapshot
        self._service.add_discrete_node(snap.name, list(snap.states))
        for parent in snap.parents:
            self._service.add_edge(parent, snap.name)
        # Restore children: re-add each edge from this node to its former children.
        for child in snap.children:
            try:
                self._service.add_edge(snap.name, child)
            except Exception:
                pass
        if snap.cpt_flat.size:
            self._service.set_flat_cpt(snap.name, snap.cpt_flat)
        self._notify()


class RenameNodeCommand(_BaseNodeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: StructureChangedCallback,
        old: str,
        new: str,
    ) -> None:
        super().__init__(f"Rename '{old}' → '{new}'", service, on_structure_changed)
        self._old = old
        self._new = new

    def redo(self) -> None:  # type: ignore[override]
        self._service.rename_node(self._old, self._new)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        self._service.rename_node(self._new, self._old)
        self._notify()


class MoveNodesCommand(QUndoCommand):
    """Move is GUI-only; it doesn't touch the service."""

    def __init__(
        self,
        positions_before: dict[str, tuple[float, float]],
        positions_after: dict[str, tuple[float, float]],
        apply: Callable[[dict[str, tuple[float, float]]], None],
    ) -> None:
        super().__init__("Move nodes")
        self._before = dict(positions_before)
        self._after = dict(positions_after)
        self._apply = apply

    def id(self) -> int:  # type: ignore[override]
        return MOVE_COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:  # type: ignore[override]
        if other.id() != self.id() or not isinstance(other, MoveNodesCommand):
            return False
        # Keep our "before" (the earliest), take their "after" (the latest).
        self._after = dict(other._after)
        return True

    def redo(self) -> None:  # type: ignore[override]
        self._apply(self._after)

    def undo(self) -> None:  # type: ignore[override]
        self._apply(self._before)
