"""QUndoCommand subclasses for node-scoped mutations."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
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


class AddNoisyMaxCommand(_BaseNodeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: StructureChangedCallback,
        name: str,
        states: Sequence[str],
        parents: Sequence[str],
        link_matrices: Mapping[str, np.ndarray],
        leak_probs: np.ndarray,
    ) -> None:
        super().__init__(f"Add Noisy-MAX '{name}'", service, on_structure_changed)
        self._name = name
        self._states = tuple(states)
        self._parents = tuple(parents)
        self._link_matrices = {k: np.asarray(v, dtype=np.float64).copy() for k, v in link_matrices.items()}
        self._leak_probs = np.asarray(leak_probs, dtype=np.float64).copy()

    def redo(self) -> None:  # type: ignore[override]
        self._service.add_noisy_max_node(
            self._name,
            self._states,
            self._parents,
            self._link_matrices,
            self._leak_probs,
        )
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        self._service.remove_node(self._name)
        self._notify()


class AddEquationNodeCommand(_BaseNodeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: StructureChangedCallback,
        name: str,
        states: Sequence[str],
        parents: Sequence[str],
        expression: Callable[..., str],
    ) -> None:
        super().__init__(f"Add equation node '{name}'", service, on_structure_changed)
        self._name = name
        self._states = tuple(states)
        self._parents = tuple(parents)
        self._expression = expression

    def redo(self) -> None:  # type: ignore[override]
        self._service.add_equation_node(
            self._name,
            self._states,
            self._parents,
            self._expression,
        )
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        self._service.remove_node(self._name)
        self._notify()


class EditStatesCommand(_BaseNodeCommand):
    def __init__(
        self,
        service: AuthoringService,
        on_structure_changed: StructureChangedCallback,
        name: str,
        new_states: Sequence[str],
    ) -> None:
        super().__init__(f"Edit states of '{name}'", service, on_structure_changed)
        self._name = name
        self._new_states = tuple(new_states)
        self._prior: NodeSnapshot | None = None
        self._prior_children_cpts: dict[str, "np.ndarray"] = {}

    def redo(self) -> None:  # type: ignore[override]
        self._prior = self._service.update_node_states(self._name, self._new_states)
        # Capture child CPTs at the moment of the first run so undo restores
        # both this node's CPT and the affected children's CPTs.
        self._prior_children_cpts = {}
        if self._prior is not None:
            for child in self._prior.children:
                child_snap = self._service.node_snapshot(child)
                if child_snap.cpt_flat.size:
                    self._prior_children_cpts[child] = child_snap.cpt_flat
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        if self._prior is None:
            return
        self._service.update_node_states(self._name, self._prior.states)
        if self._prior.cpt_flat.size:
            try:
                self._service.set_flat_cpt(self._name, self._prior.cpt_flat)
            except Exception:
                pass
        for child, flat in self._prior_children_cpts.items():
            try:
                self._service.set_flat_cpt(child, flat)
            except Exception:
                pass
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
