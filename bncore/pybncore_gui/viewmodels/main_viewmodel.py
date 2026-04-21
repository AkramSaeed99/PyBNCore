"""Central viewmodel coordinating services, workers, commands, and UI state."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QUndoStack

from pybncore_gui.commands import (
    AddEdgeCommand,
    AddNodeCommand,
    MoveNodesCommand,
    RemoveEdgeCommand,
    RemoveNodeCommand,
    RenameNodeCommand,
    SetCPTCommand,
)
from pybncore_gui.domain.errors import DomainError
from pybncore_gui.domain.node import EdgeModel, NodeModel
from pybncore_gui.domain.project import ProjectFile
from pybncore_gui.domain.results import CompileStats, PosteriorResult
from pybncore_gui.domain.session import ModelSession
from pybncore_gui.domain.validation import ValidationReport
from pybncore_gui.services.authoring_service import AuthoringService
from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.services.io_service import IOService
from pybncore_gui.services.model_service import ModelService
from pybncore_gui.services.validation_service import ValidationService
from pybncore_gui.workers.base_worker import BaseWorker
from pybncore_gui.workers.compile_worker import CompileWorker
from pybncore_gui.workers.query_worker import QueryWorker

logger = logging.getLogger(__name__)


class MainViewModel(QObject):
    """Owns evidence, undo stack, and the currently-running worker thread."""

    model_loaded = Signal(list, list)
    model_cleared = Signal()
    structure_changed = Signal(list, list)   # after add/remove/rename ops
    positions_changed = Signal(dict)          # {node_id: (x, y)}
    selection_changed = Signal(str)
    evidence_changed = Signal(dict)
    compile_started = Signal()
    compile_finished = Signal(object)
    compile_failed = Signal(str)
    query_started = Signal(str)
    query_finished = Signal(object)
    query_failed = Signal(str)
    busy_changed = Signal(bool)
    log_message = Signal(str, str)
    validation_report = Signal(object)        # ValidationReport

    def __init__(
        self,
        session: ModelSession,
        io_service: IOService,
        model_service: ModelService,
        inference_service: InferenceService,
        authoring_service: AuthoringService,
        validation_service: ValidationService,
    ) -> None:
        super().__init__()
        self._session = session
        self._io = io_service
        self._model_service = model_service
        self._inference = inference_service
        self._authoring = authoring_service
        self._validation = validation_service
        self._evidence: dict[str, str] = {}
        self._selected_node: str = ""
        self._positions: dict[str, tuple[float, float]] = {}
        self._thread: Optional[QThread] = None
        self._worker: Optional[BaseWorker] = None
        self._undo_stack = QUndoStack(self)
        self._undo_stack.setUndoLimit(200)

    # ------------------------------------------------------------------ props

    @property
    def session(self) -> ModelSession:
        return self._session

    @property
    def model_service(self) -> ModelService:
        return self._model_service

    @property
    def authoring_service(self) -> AuthoringService:
        return self._authoring

    @property
    def evidence(self) -> dict[str, str]:
        return dict(self._evidence)

    @property
    def selected_node(self) -> str:
        return self._selected_node

    @property
    def positions(self) -> dict[str, tuple[float, float]]:
        return dict(self._positions)

    @property
    def undo_stack(self) -> QUndoStack:
        return self._undo_stack

    @property
    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    # ------------------------------------------------------------------ IO

    @Slot(str)
    def open_xdsl(self, path: str) -> None:
        try:
            self._io.open_xdsl(path)
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self._positions.clear()
        self._on_model_changed(source=f"Loaded {path}")

    @Slot(str)
    def import_bif(self, path: str) -> None:
        try:
            self._io.import_bif(path)
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self._positions.clear()
        self._on_model_changed(source=f"Imported BIF {path}")

    @Slot(str)
    def save_xdsl(self, path: str) -> None:
        try:
            self._io.save_xdsl(path)
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self.log_message.emit("info", f"Saved XDSL → {path}")

    @Slot(str)
    def save_project(self, path: str) -> None:
        try:
            self._io.save_project(path, self._positions)
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self.log_message.emit("info", f"Saved project → {path}")

    @Slot(str)
    def load_project(self, path: str) -> None:
        try:
            project = self._io.load_project(path)
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self._positions = dict(project.positions)
        self._on_model_changed(source=f"Loaded project {path}")
        if self._positions:
            self.positions_changed.emit(self.positions)

    @Slot()
    def new_model(self) -> None:
        try:
            self._io.new_empty()
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self._positions.clear()
        self._on_model_changed(source="New empty model")

    # ------------------------------------------------------------ authoring

    @Slot(str, list, list)
    def add_discrete_node(
        self,
        name: str,
        states: Sequence[str],
        parents: Sequence[str] = (),
    ) -> None:
        cmd = AddNodeCommand(
            self._authoring, self._emit_structure, name, states, parents
        )
        self._push(cmd)

    @Slot(str)
    def remove_node(self, name: str) -> None:
        cmd = RemoveNodeCommand(self._authoring, self._emit_structure, name)
        self._push(cmd)

    @Slot(str, str)
    def rename_node(self, old: str, new: str) -> None:
        if old == new:
            return
        cmd = RenameNodeCommand(self._authoring, self._emit_structure, old, new)
        self._push(cmd)

    @Slot(str, str)
    def add_edge(self, parent: str, child: str) -> None:
        cmd = AddEdgeCommand(self._authoring, self._emit_structure, parent, child)
        self._push(cmd)

    @Slot(str, str)
    def remove_edge(self, parent: str, child: str) -> None:
        cmd = RemoveEdgeCommand(self._authoring, self._emit_structure, parent, child)
        self._push(cmd)

    @Slot(str, object)
    def set_cpt(self, node: str, shaped: np.ndarray) -> None:
        cmd = SetCPTCommand(self._authoring, self._emit_structure, node, shaped)
        self._push(cmd)

    @Slot(dict, dict)
    def record_move(
        self,
        positions_before: dict[str, tuple[float, float]],
        positions_after: dict[str, tuple[float, float]],
    ) -> None:
        if positions_before == positions_after:
            return
        cmd = MoveNodesCommand(positions_before, positions_after, self._apply_positions)
        self._push(cmd)

    # ----------------------------------------------------------- validation

    @Slot()
    def validate(self) -> None:
        try:
            report = self._validation.validate()
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self.validation_report.emit(report)
        errors = sum(1 for i in report.issues if i.severity.value == "error")
        warnings = sum(1 for i in report.issues if i.severity.value == "warning")
        self.log_message.emit(
            "info",
            f"Validation: {len(report.issues)} issue(s) "
            f"({errors} errors, {warnings} warnings).",
        )

    # ---------------------------------------------------------- UI state

    @Slot(str)
    def select_node(self, node_id: str) -> None:
        self._selected_node = node_id or ""
        self.selection_changed.emit(self._selected_node)

    @Slot(str, object)
    def set_evidence(self, node_id: str, state: Optional[str]) -> None:
        if not node_id:
            return
        if state is None or state == "":
            self._evidence.pop(node_id, None)
        else:
            self._evidence[node_id] = state
        self.evidence_changed.emit(dict(self._evidence))

    @Slot()
    def clear_evidence(self) -> None:
        if not self._evidence:
            return
        self._evidence.clear()
        self.evidence_changed.emit({})

    @Slot(str, float, float)
    def update_position(self, node_id: str, x: float, y: float) -> None:
        self._positions[node_id] = (float(x), float(y))

    # ----------------------------------------------------------- inference

    @Slot()
    def compile(self) -> None:
        if not self._guard_busy_or_empty():
            return
        self.compile_started.emit()
        worker = CompileWorker(self._inference)
        self._run_worker(worker, self._on_compile_finished, self._on_compile_failed)

    @Slot(str)
    def run_query(self, node_id: Optional[str] = None) -> None:
        target = node_id or self._selected_node
        if not target:
            self.log_message.emit("warning", "Select a node before running a query.")
            return
        if not self._guard_busy_or_empty():
            return
        self.query_started.emit(target)
        worker = QueryWorker(self._inference, target, self._evidence)
        self._run_worker(worker, self._on_query_finished, self._on_query_failed)

    # --------------------------------------------------------------- internals

    def _guard_busy_or_empty(self) -> bool:
        if self.is_busy:
            self.log_message.emit("warning", "Another operation is already running.")
            return False
        if not self._session.has_model:
            self.log_message.emit("warning", "No model is loaded.")
            return False
        return True

    def _push(self, command) -> None:
        try:
            self._undo_stack.push(command)
        except DomainError as e:
            self.log_message.emit("error", e.user_message)

    def _emit_structure(self) -> None:
        nodes = self._model_service.list_nodes()
        edges = self._model_service.list_edges()
        # Drop positions for removed nodes
        alive = {n.id for n in nodes}
        for dead in [k for k in self._positions if k not in alive]:
            self._positions.pop(dead, None)
        # Drop evidence for removed / renamed nodes
        for dead in [k for k in self._evidence if k not in alive]:
            self._evidence.pop(dead, None)
        self.structure_changed.emit(nodes, edges)
        self.evidence_changed.emit(dict(self._evidence))
        self.log_message.emit("info", f"Structure updated — {len(nodes)} nodes, {len(edges)} edges.")

    def _on_model_changed(self, source: str) -> None:
        self._evidence.clear()
        self._selected_node = ""
        self._undo_stack.clear()
        nodes = self._model_service.list_nodes()
        edges = self._model_service.list_edges()
        self.evidence_changed.emit({})
        self.selection_changed.emit("")
        self.model_loaded.emit(nodes, edges)
        self.log_message.emit(
            "info",
            f"{source} — {len(nodes)} nodes, {len(edges)} edges.",
        )

    def _apply_positions(self, positions: dict[str, tuple[float, float]]) -> None:
        self._positions.update(positions)
        self.positions_changed.emit(dict(positions))

    def _run_worker(self, worker: BaseWorker, on_finished, on_failed) -> None:
        thread = QThread(self)
        worker.setParent(None)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self.busy_changed.emit(True)
        thread.start()

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.busy_changed.emit(False)

    @Slot(object)
    def _on_compile_finished(self, stats: CompileStats) -> None:
        self.compile_finished.emit(stats)
        self.log_message.emit(
            "info",
            f"Compile ok — {stats.node_count} nodes, {stats.edge_count} edges.",
        )

    @Slot(str, str)
    def _on_compile_failed(self, message: str, _tb: str) -> None:
        self.compile_failed.emit(message)
        self.log_message.emit("error", f"Compile failed: {message}")

    @Slot(object)
    def _on_query_finished(self, result: PosteriorResult) -> None:
        self.query_finished.emit(result)
        self.log_message.emit("info", f"Query on '{result.node}' completed.")

    @Slot(str, str)
    def _on_query_failed(self, message: str, _tb: str) -> None:
        self.query_failed.emit(message)
        self.log_message.emit("error", f"Query failed: {message}")
