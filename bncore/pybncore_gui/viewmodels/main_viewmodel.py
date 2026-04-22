"""Central viewmodel coordinating services, workers, commands, and UI state."""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QUndoStack

from pybncore_gui.commands import (
    AddContinuousNodeCommand,
    AddEdgeCommand,
    AddEquationNodeCommand,
    AddNodeCommand,
    AddNoisyMaxCommand,
    EditStatesCommand,
    MoveNodesCommand,
    RemoveEdgeCommand,
    RemoveNodeCommand,
    RenameNodeCommand,
    SetCPTCommand,
)
from pybncore_gui.domain.continuous import ContinuousNodeSpec
from pybncore_gui.domain.errors import DomainError
from pybncore_gui.domain.node import EdgeModel, NodeModel
from pybncore_gui.domain.project import ProjectFile
from pybncore_gui.domain.results import (
    BatchQueryResult,
    BenchmarkResult,
    CompileStats,
    HybridResultDTO,
    JTStats,
    MAPResult,
    MonteCarloResult,
    PosteriorResult,
    SensitivityReport,
    VOIReport,
)
from pybncore_gui.domain.scenario import Scenario
from pybncore_gui.domain.session import ModelSession
from pybncore_gui.domain.settings import EngineSettings
from pybncore_gui.domain.submodel import ROOT_ID, SubModelLayout
from pybncore_gui.domain.validation import ValidationReport
from pybncore_gui.services.analysis_service import AnalysisService
from pybncore_gui.services.authoring_service import AuthoringService
from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.services.io_service import IOService
from pybncore_gui.services.model_service import ModelService
from pybncore_gui.services.submodel_service import SubModelService
from pybncore_gui.services.validation_service import ValidationService
from pybncore_gui.workers.analysis_workers import SensitivityWorker, VOIWorker
from pybncore_gui.workers.base_worker import BaseWorker
from pybncore_gui.workers.batch_worker import BatchQueryWorker
from pybncore_gui.workers.benchmark_worker import BenchmarkWorker
from pybncore_gui.workers.compile_worker import CompileWorker
from pybncore_gui.workers.hybrid_worker import HybridQueryWorker
from pybncore_gui.workers.load_model_worker import LoadModelResult, LoadModelWorker
from pybncore_gui.workers.map_worker import MAPQueryWorker
from pybncore_gui.workers.monte_carlo_worker import MonteCarloWorker
from pybncore_gui.workers.query_worker import QueryWorker

logger = logging.getLogger(__name__)

_POSTERIOR_HISTORY = 20


class MainViewModel(QObject):
    """Owns evidence, soft evidence, undo stack, scenarios, and worker lifecycle."""

    model_loaded = Signal(list, list)
    model_cleared = Signal()
    structure_changed = Signal(list, list)
    positions_changed = Signal(dict)
    selection_changed = Signal(str)
    evidence_changed = Signal(dict)
    soft_evidence_changed = Signal(dict)
    compile_started = Signal()
    compile_finished = Signal(object)
    compile_failed = Signal(str)
    query_started = Signal(str)
    query_finished = Signal(object)
    query_failed = Signal(str)
    map_started = Signal()
    map_finished = Signal(object)
    map_failed = Signal(str)
    batch_started = Signal()
    batch_finished = Signal(object)
    batch_failed = Signal(str)
    sensitivity_started = Signal()
    sensitivity_finished = Signal(object)
    sensitivity_failed = Signal(str)
    voi_started = Signal()
    voi_finished = Signal(object)
    voi_failed = Signal(str)
    benchmark_started = Signal()
    benchmark_finished = Signal(object)
    benchmark_failed = Signal(str)
    monte_carlo_started = Signal()
    monte_carlo_finished = Signal(object)
    monte_carlo_failed = Signal(str)
    busy_changed = Signal(bool)
    log_message = Signal(str, str)
    validation_report = Signal(object)
    scenarios_changed = Signal(list)
    settings_changed = Signal(object)
    jt_stats_updated = Signal(object)
    posterior_history_changed = Signal(list)
    layout_changed = Signal(object)               # SubModelLayout
    current_submodel_changed = Signal(str)
    descriptions_changed = Signal(dict)
    continuous_nodes_changed = Signal(list)
    continuous_evidence_changed = Signal(dict)
    continuous_likelihoods_changed = Signal(dict)
    hybrid_started = Signal()
    hybrid_finished = Signal(object)
    hybrid_failed = Signal(str)

    def __init__(
        self,
        session: ModelSession,
        io_service: IOService,
        model_service: ModelService,
        inference_service: InferenceService,
        authoring_service: AuthoringService,
        validation_service: ValidationService,
        analysis_service: AnalysisService,
    ) -> None:
        super().__init__()
        self._session = session
        self._io = io_service
        self._model_service = model_service
        self._inference = inference_service
        self._authoring = authoring_service
        self._validation = validation_service
        self._analysis = analysis_service
        self._evidence: dict[str, str] = {}
        self._soft_evidence: dict[str, dict[str, float]] = {}
        self._scenarios: list[Scenario] = []
        self._posterior_history: deque[PosteriorResult] = deque(maxlen=_POSTERIOR_HISTORY)
        self._selected_node: str = ""
        self._positions: dict[str, tuple[float, float]] = {}
        self._layout: SubModelLayout = SubModelLayout()
        self._current_submodel: str = ROOT_ID
        self._submodel_service = SubModelService()
        self._descriptions: dict[str, str] = {}
        self._continuous_nodes: list[str] = []
        self._continuous_evidence: dict[str, float] = {}
        self._continuous_likelihoods: dict[str, object] = {}
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
    def soft_evidence(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self._soft_evidence.items()}

    @property
    def scenarios(self) -> list[Scenario]:
        return list(self._scenarios)

    @property
    def settings(self) -> EngineSettings:
        return self._inference.settings

    @property
    def posterior_history(self) -> list[PosteriorResult]:
        return list(self._posterior_history)

    @property
    def selected_node(self) -> str:
        return self._selected_node

    @property
    def positions(self) -> dict[str, tuple[float, float]]:
        return dict(self._positions)

    @property
    def layout(self) -> SubModelLayout:
        return self._layout

    @property
    def current_submodel(self) -> str:
        return self._current_submodel

    @property
    def descriptions(self) -> dict[str, str]:
        return dict(self._descriptions)

    @property
    def continuous_nodes(self) -> list[str]:
        return list(self._continuous_nodes)

    @property
    def continuous_evidence(self) -> dict[str, float]:
        return dict(self._continuous_evidence)

    @property
    def continuous_likelihoods(self) -> dict[str, object]:
        return dict(self._continuous_likelihoods)

    @property
    def undo_stack(self) -> QUndoStack:
        return self._undo_stack

    @property
    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    # ------------------------------------------------------------------ IO

    @Slot(str)
    def open_xdsl(self, path: str) -> None:
        self._start_load(path, kind="xdsl")

    @Slot(str)
    def import_bif(self, path: str) -> None:
        self._start_load(path, kind="bif")

    def _start_load(self, path: str, *, kind: str) -> None:
        if self.is_busy:
            self.log_message.emit(
                "warning", "Another operation is running — try again in a moment."
            )
            return
        worker = LoadModelWorker(self._io, path, kind=kind)
        self.log_message.emit("info", f"Loading {path}…")
        self._run_worker(
            worker,
            self._on_model_load_finished,
            self._on_model_load_failed,
        )

    @Slot(object)
    def _on_model_load_finished(self, result: LoadModelResult) -> None:
        self._positions.clear()
        if result.kind == "xdsl":
            self._load_layout_from_xdsl(result.path)
            source = f"Loaded {result.path}"
        else:
            source = f"Imported BIF {result.path}"
        self._on_model_changed(source=source)

    @Slot(str, str)
    def _on_model_load_failed(self, message: str, _tb: str) -> None:
        self.log_message.emit("error", message)

    @Slot(str)
    def save_xdsl(self, path: str) -> None:
        try:
            self._submodel_service.sync_nodes(
                self._layout, [n.id for n in self._model_service.list_nodes()]
            )
            self._io.save_xdsl(
                path,
                layout=self._layout,
                node_positions=self._positions,
                descriptions=self._descriptions,
            )
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self.log_message.emit("info", f"Saved XDSL → {path}")

    @Slot(str)
    def save_project(self, path: str) -> None:
        try:
            self._submodel_service.sync_nodes(
                self._layout, [n.id for n in self._model_service.list_nodes()]
            )
            self._io.save_project(
                path,
                self._positions,
                scenarios=[s.to_dict() for s in self._scenarios],
                settings=self.settings,
                layout=self._layout,
                descriptions=self._descriptions,
            )
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
        self._scenarios = list(project.scenarios)
        self._layout = project.layout
        self._descriptions = dict(project.descriptions)
        self._current_submodel = ROOT_ID
        self._inference.update_settings(project.settings)
        self._on_model_changed(source=f"Loaded project {path}")
        if self._positions:
            self.positions_changed.emit(self.positions)
        self.scenarios_changed.emit(self.scenarios)
        self.settings_changed.emit(self.settings)
        self.descriptions_changed.emit(self.descriptions)

    @Slot()
    def new_model(self) -> None:
        try:
            self._io.new_empty()
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self._positions.clear()
        self._scenarios.clear()
        self._layout = SubModelLayout()
        self._descriptions.clear()
        self._current_submodel = ROOT_ID
        self._on_model_changed(source="New empty model")
        self.scenarios_changed.emit(self.scenarios)
        self.descriptions_changed.emit({})

    def _load_layout_from_xdsl(self, path: str) -> None:
        try:
            self._layout = self._io.parse_submodel_layout(path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to parse submodel layout from %s", path)
            self._layout = SubModelLayout()
        try:
            self._descriptions = self._io.submodel_service.parse_descriptions(path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to parse descriptions from %s", path)
            self._descriptions = {}
        self._current_submodel = ROOT_ID
        # Extract node positions from the parsed XDSL (not touched by save_xdsl fallback).
        from xml.etree import ElementTree as ET
        try:
            tree = ET.parse(path)
        except Exception:
            return
        genie = tree.getroot().find(".//extensions/genie")
        if genie is None:
            return

        def walk(el):
            for node_el in el.findall("./node"):
                nid = node_el.get("id")
                pos_el = node_el.find("position")
                if nid and pos_el is not None and pos_el.text:
                    try:
                        nums = [float(x) for x in pos_el.text.split()]
                        if len(nums) >= 2:
                            self._positions[nid] = (nums[0], nums[1])
                    except ValueError:
                        pass
            for sm_el in el.findall("./submodel"):
                walk(sm_el)

        walk(genie)

    # ------------------------------------------------------------ authoring

    @Slot(str, list, list)
    def add_discrete_node(
        self,
        name: str,
        states: Sequence[str],
        parents: Sequence[str] = (),
    ) -> None:
        cmd = AddNodeCommand(self._authoring, self._emit_structure, name, states, parents)
        self._push(cmd)

    @Slot(str)
    def remove_node(self, name: str) -> None:
        self._push(RemoveNodeCommand(self._authoring, self._emit_structure, name))

    @Slot(str, str)
    def rename_node(self, old: str, new: str) -> None:
        if old == new:
            return
        self._push(RenameNodeCommand(self._authoring, self._emit_structure, old, new))

    @Slot(str, list)
    def edit_node_states(self, name: str, new_states: Sequence[str]) -> None:
        self._push(
            EditStatesCommand(
                self._authoring, self._emit_structure, name, tuple(new_states)
            )
        )

    @Slot(str, str)
    def add_edge(self, parent: str, child: str) -> None:
        self._push(AddEdgeCommand(self._authoring, self._emit_structure, parent, child))

    @Slot(str, str)
    def remove_edge(self, parent: str, child: str) -> None:
        self._push(RemoveEdgeCommand(self._authoring, self._emit_structure, parent, child))

    @Slot(str, object)
    def set_cpt(self, node: str, shaped: np.ndarray) -> None:
        self._push(SetCPTCommand(self._authoring, self._emit_structure, node, shaped))

    @Slot(str, list, list, object, object)
    def add_noisy_max_node(
        self,
        name: str,
        states: Sequence[str],
        parents: Sequence[str],
        link_matrices: Mapping[str, np.ndarray],
        leak_probs: np.ndarray,
    ) -> None:
        cmd = AddNoisyMaxCommand(
            self._authoring,
            self._emit_structure,
            name,
            states,
            parents,
            link_matrices,
            leak_probs,
        )
        self._push(cmd)

    @Slot(object)
    def add_continuous_node(self, spec: ContinuousNodeSpec) -> None:
        cmd = AddContinuousNodeCommand(
            self._authoring, self._emit_structure, spec
        )
        self._push(cmd)

    @Slot(str, float)
    def add_threshold(self, node: str, threshold: float) -> None:
        try:
            self._authoring.add_threshold(node, float(threshold))
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self.log_message.emit(
            "info", f"Threshold {threshold:.6g} added on '{node}'."
        )
        self._emit_structure()

    @Slot(str, object)
    def set_continuous_value(self, node: str, value) -> None:
        if not node:
            return
        if value is None:
            self._continuous_evidence.pop(node, None)
        else:
            self._continuous_evidence[node] = float(value)
            # A hard value supersedes a likelihood on the same node.
            self._continuous_likelihoods.pop(node, None)
        self.continuous_evidence_changed.emit(self.continuous_evidence)
        self.continuous_likelihoods_changed.emit(self.continuous_likelihoods)

    @Slot(str, object)
    def set_continuous_likelihood(self, node: str, fn) -> None:
        if not node:
            return
        if fn is None:
            self._continuous_likelihoods.pop(node, None)
        else:
            self._continuous_likelihoods[node] = fn
            self._continuous_evidence.pop(node, None)
        self.continuous_evidence_changed.emit(self.continuous_evidence)
        self.continuous_likelihoods_changed.emit(self.continuous_likelihoods)

    @Slot()
    def clear_all_continuous_evidence(self) -> None:
        self._continuous_evidence.clear()
        self._continuous_likelihoods.clear()
        self.continuous_evidence_changed.emit({})
        self.continuous_likelihoods_changed.emit({})

    @Slot()
    def run_hybrid(self) -> None:
        if not self._guard_busy_or_empty():
            return
        if not self._continuous_nodes:
            self.log_message.emit(
                "warning", "No continuous nodes registered — add one first."
            )
            return
        self.hybrid_started.emit()
        worker = HybridQueryWorker(
            self._inference,
            list(self._continuous_nodes),
            evidence=self._evidence,
            soft_evidence=self._soft_evidence,
            continuous_evidence=self._continuous_evidence,
            continuous_likelihoods=self._continuous_likelihoods,
        )
        self._run_worker(worker, self._on_hybrid_finished, self._on_hybrid_failed)

    @Slot(str, list, list, object)
    def add_equation_node(
        self,
        name: str,
        states: Sequence[str],
        parents: Sequence[str],
        expression,
    ) -> None:
        cmd = AddEquationNodeCommand(
            self._authoring,
            self._emit_structure,
            name,
            states,
            parents,
            expression,
        )
        self._push(cmd)

    @Slot(dict, dict)
    def record_move(
        self,
        positions_before: dict[str, tuple[float, float]],
        positions_after: dict[str, tuple[float, float]],
    ) -> None:
        if positions_before == positions_after:
            return
        self._push(MoveNodesCommand(positions_before, positions_after, self._apply_positions))

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
            f"Validation: {len(report.issues)} issue(s) ({errors} errors, {warnings} warnings).",
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

    @Slot(str, dict)
    def set_soft_evidence(
        self, node: str, likelihoods: Mapping[str, float] | None
    ) -> None:
        if not node:
            return
        if not likelihoods:
            self._soft_evidence.pop(node, None)
        else:
            self._soft_evidence[node] = {str(k): float(v) for k, v in likelihoods.items()}
        self.soft_evidence_changed.emit(self.soft_evidence)

    @Slot()
    def clear_soft_evidence(self) -> None:
        if not self._soft_evidence:
            return
        self._soft_evidence.clear()
        self.soft_evidence_changed.emit({})

    @Slot(str, float, float)
    def update_position(self, node_id: str, x: float, y: float) -> None:
        self._positions[node_id] = (float(x), float(y))

    @Slot(str, str)
    def set_description(self, node_id: str, description: str) -> None:
        if not node_id:
            return
        cleaned = description.strip()
        if cleaned:
            self._descriptions[node_id] = cleaned
        else:
            self._descriptions.pop(node_id, None)
        self.descriptions_changed.emit(self.descriptions)
        # Re-emit structure so the canvas picks up the new description.
        nodes = self._model_service.list_nodes()
        edges = self._model_service.list_edges()
        self.structure_changed.emit(nodes, edges)

    @Slot(str, float, float)
    def update_submodel_position(self, submodel_id: str, x: float, y: float) -> None:
        sm = self._layout.submodels.get(submodel_id)
        if sm is None:
            return
        l, t, r, b = sm.position
        w = max(120.0, r - l)
        h = max(80.0, b - t)
        sm.position = (float(x), float(y), float(x + w), float(y + h))

    # ----------------------------------------------------------- sub-models

    @Slot(str)
    def enter_submodel(self, submodel_id: str) -> None:
        if submodel_id != ROOT_ID and submodel_id not in self._layout.submodels:
            return
        self._current_submodel = submodel_id
        self.current_submodel_changed.emit(submodel_id)
        self._emit_view()

    @Slot()
    def exit_submodel(self) -> None:
        if self._current_submodel == ROOT_ID:
            return
        parent = self._layout.parent_of(self._current_submodel, is_submodel=True)
        self.enter_submodel(parent)

    @Slot(str, str)
    def create_submodel(self, name: str, parent_id: str | None = None) -> None:
        parent = parent_id if parent_id is not None else self._current_submodel
        sm = self._submodel_service.create_submodel(self._layout, name, parent)
        self.layout_changed.emit(self._layout)
        self.log_message.emit("info", f"Created sub-model '{sm.name}'.")
        self._emit_view()

    @Slot(str)
    def delete_submodel(self, submodel_id: str) -> None:
        if submodel_id == ROOT_ID:
            return
        if self._current_submodel == submodel_id:
            self._current_submodel = self._layout.parent_of(submodel_id, is_submodel=True)
            self.current_submodel_changed.emit(self._current_submodel)
        self._submodel_service.delete_submodel(self._layout, submodel_id)
        self.layout_changed.emit(self._layout)
        self.log_message.emit("info", f"Deleted sub-model '{submodel_id}'.")
        self._emit_view()

    @Slot(str, str)
    def rename_submodel(self, submodel_id: str, new_name: str) -> None:
        self._submodel_service.rename_submodel(self._layout, submodel_id, new_name)
        self.layout_changed.emit(self._layout)
        self._emit_view()

    @Slot(list, str)
    def move_nodes_to_submodel(self, node_ids: Sequence[str], target_submodel: str) -> None:
        self._submodel_service.move_nodes(self._layout, node_ids, target_submodel)
        self.layout_changed.emit(self._layout)
        self.log_message.emit(
            "info",
            f"Moved {len(node_ids)} node(s) to "
            f"'{target_submodel or 'Root'}'.",
        )
        self._emit_view()

    def _emit_view(self) -> None:
        """Notify views that the current-submodel scoping may have changed."""
        nodes = self._model_service.list_nodes()
        edges = self._model_service.list_edges()
        self.structure_changed.emit(nodes, edges)

    # ----------------------------------------------------------- scenarios

    @Slot(str)
    def save_current_scenario(self, name: str) -> None:
        if not name:
            self.log_message.emit("warning", "Scenario name is required.")
            return
        scenario = Scenario(
            name=name,
            evidence=dict(self._evidence),
            soft_evidence={k: dict(v) for k, v in self._soft_evidence.items()},
        )
        # Replace if name already exists.
        for i, existing in enumerate(self._scenarios):
            if existing.name == name:
                self._scenarios[i] = scenario
                break
        else:
            self._scenarios.append(scenario)
        self.scenarios_changed.emit(self.scenarios)
        self.log_message.emit("info", f"Saved scenario '{name}'.")

    @Slot(str)
    def apply_scenario(self, name: str) -> None:
        scenario = next((s for s in self._scenarios if s.name == name), None)
        if scenario is None:
            self.log_message.emit("warning", f"Scenario '{name}' not found.")
            return
        self._evidence = dict(scenario.evidence)
        self._soft_evidence = {k: dict(v) for k, v in scenario.soft_evidence.items()}
        self.evidence_changed.emit(self.evidence)
        self.soft_evidence_changed.emit(self.soft_evidence)
        self.log_message.emit("info", f"Applied scenario '{name}'.")

    @Slot(str)
    def delete_scenario(self, name: str) -> None:
        before = len(self._scenarios)
        self._scenarios = [s for s in self._scenarios if s.name != name]
        if len(self._scenarios) != before:
            self.scenarios_changed.emit(self.scenarios)
            self.log_message.emit("info", f"Deleted scenario '{name}'.")

    @Slot(str, str)
    def rename_scenario(self, old: str, new: str) -> None:
        if not new or old == new:
            return
        if any(s.name == new for s in self._scenarios):
            self.log_message.emit("warning", f"Scenario '{new}' already exists.")
            return
        for scenario in self._scenarios:
            if scenario.name == old:
                scenario.name = new
                self.scenarios_changed.emit(self.scenarios)
                return

    # ------------------------------------------------------------- settings

    @Slot(object)
    def update_settings(self, settings: EngineSettings) -> None:
        self._inference.update_settings(settings)
        self.settings_changed.emit(self.settings)
        self.log_message.emit(
            "info",
            f"Engine settings updated (loopy={self.settings.use_loopy_bp}, "
            f"heuristic={self.settings.triangulation}).",
        )

    @Slot()
    def fetch_jt_stats(self) -> None:
        if not self._session.has_model:
            return
        try:
            stats = self._inference.compute_jt_stats()
        except DomainError as e:
            self.log_message.emit("error", e.user_message)
            return
        self.jt_stats_updated.emit(stats)

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
        worker = QueryWorker(self._inference, target, self._evidence, self._soft_evidence)
        self._run_worker(worker, self._on_query_finished, self._on_query_failed)

    @Slot()
    def run_map(self) -> None:
        if not self._guard_busy_or_empty():
            return
        self.map_started.emit()
        worker = MAPQueryWorker(self._inference, self._evidence, self._soft_evidence)
        self._run_worker(worker, self._on_map_finished, self._on_map_failed)

    @Slot(str, str, int, float)
    def run_sensitivity(
        self, query_node: str, query_state: str, n_top: int, epsilon: float
    ) -> None:
        if not self._guard_busy_or_empty():
            return
        self.sensitivity_started.emit()
        worker = SensitivityWorker(
            self._analysis,
            query_node,
            query_state,
            n_top,
            epsilon,
            self._evidence,
            self._soft_evidence,
        )
        self._run_worker(worker, self._on_sensitivity_finished, self._on_sensitivity_failed)

    @Slot(str, list)
    def run_voi(self, query_node: str, candidates: Sequence[str] | None = None) -> None:
        if not self._guard_busy_or_empty():
            return
        self.voi_started.emit()
        worker = VOIWorker(
            self._analysis,
            query_node,
            candidates,
            self._evidence,
            self._soft_evidence,
        )
        self._run_worker(worker, self._on_voi_finished, self._on_voi_failed)

    @Slot(list, list, list, object)
    def run_benchmark(
        self,
        query_nodes: Sequence[str],
        observed_nodes: Sequence[str],
        row_counts: Sequence[int],
        seed: int | None = None,
    ) -> None:
        if not self._guard_busy_or_empty():
            return
        self.benchmark_started.emit()
        worker = BenchmarkWorker(
            self._analysis, query_nodes, observed_nodes, row_counts, seed
        )
        self._run_worker(worker, self._on_benchmark_finished, self._on_benchmark_failed)

    @Slot(list, list, int, object)
    def run_monte_carlo(
        self,
        query_nodes: Sequence[str],
        observed_nodes: Sequence[str],
        num_samples: int,
        seed: int | None = None,
    ) -> None:
        if not self._guard_busy_or_empty():
            return
        self.monte_carlo_started.emit()
        worker = MonteCarloWorker(
            self._analysis, query_nodes, observed_nodes, num_samples, seed
        )
        self._run_worker(worker, self._on_monte_carlo_finished, self._on_monte_carlo_failed)

    @Slot(list, list)
    def run_batch(
        self,
        query_nodes: Sequence[str],
        evidence_rows: Sequence[Mapping[str, str]],
    ) -> None:
        if not self._guard_busy_or_empty():
            return
        self.batch_started.emit()
        worker = BatchQueryWorker(self._inference, list(query_nodes), list(evidence_rows))
        self._run_worker(worker, self._on_batch_finished, self._on_batch_failed)

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
        alive = {n.id for n in nodes}
        for dead in [k for k in self._positions if k not in alive]:
            self._positions.pop(dead, None)
        for dead in [k for k in self._evidence if k not in alive]:
            self._evidence.pop(dead, None)
        for dead in [k for k in self._soft_evidence if k not in alive]:
            self._soft_evidence.pop(dead, None)
        for dead in [k for k in self._descriptions if k not in alive]:
            self._descriptions.pop(dead, None)
        # Keep the sub-model layout in sync with the live node set.
        self._submodel_service.sync_nodes(self._layout, alive)
        # Refresh the cached list of continuous nodes (wrapper knows best).
        self._refresh_continuous_nodes()
        self.structure_changed.emit(nodes, edges)
        self.layout_changed.emit(self._layout)
        self.evidence_changed.emit(dict(self._evidence))
        self.soft_evidence_changed.emit(self.soft_evidence)
        self.log_message.emit("info", f"Structure updated — {len(nodes)} nodes, {len(edges)} edges.")

    def _on_model_changed(self, source: str) -> None:
        self._evidence.clear()
        self._soft_evidence.clear()
        self._selected_node = ""
        self._undo_stack.clear()
        self._posterior_history.clear()
        nodes = self._model_service.list_nodes()
        edges = self._model_service.list_edges()
        # Rehome any unknown nodes to root, drop stale entries.
        self._submodel_service.sync_nodes(self._layout, {n.id for n in nodes})
        self.evidence_changed.emit({})
        self.soft_evidence_changed.emit({})
        self.selection_changed.emit("")
        self.layout_changed.emit(self._layout)
        self.current_submodel_changed.emit(self._current_submodel)
        self._refresh_continuous_nodes()
        self.continuous_evidence_changed.emit({})
        self.continuous_likelihoods_changed.emit({})
        self.model_loaded.emit(nodes, edges)
        self.posterior_history_changed.emit([])
        self.log_message.emit(
            "info",
            f"{source} — {len(nodes)} nodes, {len(edges)} edges.",
        )

    def _refresh_continuous_nodes(self) -> None:
        session = self._session
        names: list[str] = []
        with session.locked() as wrapper:
            if wrapper is not None:
                raw = getattr(wrapper, "_continuous_names", None) or []
                names = [str(n) for n in raw]
        # Drop stale evidence / likelihoods.
        alive = set(names)
        for dead in [k for k in self._continuous_evidence if k not in alive]:
            self._continuous_evidence.pop(dead, None)
        for dead in [k for k in self._continuous_likelihoods if k not in alive]:
            self._continuous_likelihoods.pop(dead, None)
        if names != self._continuous_nodes:
            self._continuous_nodes = names
            self.continuous_nodes_changed.emit(list(names))

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
        # Refresh JT stats for the settings dialog and any listeners.
        try:
            self.jt_stats_updated.emit(self._inference.compute_jt_stats())
        except DomainError:
            pass

    @Slot(str, str)
    def _on_compile_failed(self, message: str, _tb: str) -> None:
        self.compile_failed.emit(message)
        self.log_message.emit("error", f"Compile failed: {message}")

    @Slot(object)
    def _on_query_finished(self, result: PosteriorResult) -> None:
        self._posterior_history.append(result)
        self.query_finished.emit(result)
        self.posterior_history_changed.emit(list(self._posterior_history))
        self.log_message.emit("info", f"Query on '{result.node}' completed ({result.engine_label}).")

    @Slot(str, str)
    def _on_query_failed(self, message: str, _tb: str) -> None:
        self.query_failed.emit(message)
        self.log_message.emit("error", f"Query failed: {message}")

    @Slot(object)
    def _on_map_finished(self, result: MAPResult) -> None:
        self.map_finished.emit(result)
        self.log_message.emit("info", f"MAP completed — {len(result.assignment)} nodes.")

    @Slot(str, str)
    def _on_map_failed(self, message: str, _tb: str) -> None:
        self.map_failed.emit(message)
        self.log_message.emit("error", f"MAP failed: {message}")

    @Slot(object)
    def _on_batch_finished(self, result: BatchQueryResult) -> None:
        self.batch_finished.emit(result)
        self.log_message.emit(
            "info",
            f"Batch query completed — {result.num_rows} rows × {len(result.nodes)} query nodes.",
        )

    @Slot(str, str)
    def _on_batch_failed(self, message: str, _tb: str) -> None:
        self.batch_failed.emit(message)
        self.log_message.emit("error", f"Batch query failed: {message}")

    @Slot(object)
    def _on_sensitivity_finished(self, report: SensitivityReport) -> None:
        self.sensitivity_finished.emit(report)
        self.log_message.emit(
            "info",
            f"Sensitivity: {len(report.entries)} entries for "
            f"P({report.query_node}={report.query_state}).",
        )

    @Slot(str, str)
    def _on_sensitivity_failed(self, message: str, _tb: str) -> None:
        self.sensitivity_failed.emit(message)
        self.log_message.emit("error", f"Sensitivity failed: {message}")

    @Slot(object)
    def _on_voi_finished(self, report: VOIReport) -> None:
        self.voi_finished.emit(report)
        self.log_message.emit(
            "info",
            f"VOI completed for '{report.query_node}' — {len(report.entries)} candidates.",
        )

    @Slot(str, str)
    def _on_voi_failed(self, message: str, _tb: str) -> None:
        self.voi_failed.emit(message)
        self.log_message.emit("error", f"VOI failed: {message}")

    @Slot(object)
    def _on_benchmark_finished(self, result: BenchmarkResult) -> None:
        self.benchmark_finished.emit(result)
        self.log_message.emit(
            "info",
            f"Benchmark: {len(result.points)} size(s) over "
            f"{len(result.query_nodes)} query nodes.",
        )

    @Slot(str, str)
    def _on_benchmark_failed(self, message: str, _tb: str) -> None:
        self.benchmark_failed.emit(message)
        self.log_message.emit("error", f"Benchmark failed: {message}")

    @Slot(object)
    def _on_monte_carlo_finished(self, result: MonteCarloResult) -> None:
        self.monte_carlo_finished.emit(result)
        self.log_message.emit(
            "info",
            f"Monte Carlo: {result.num_samples} samples over "
            f"{len(result.query_nodes)} query nodes.",
        )

    @Slot(str, str)
    def _on_monte_carlo_failed(self, message: str, _tb: str) -> None:
        self.monte_carlo_failed.emit(message)
        self.log_message.emit("error", f"Monte Carlo failed: {message}")

    @Slot(object)
    def _on_hybrid_finished(self, result: HybridResultDTO) -> None:
        self.hybrid_finished.emit(result)
        self.log_message.emit(
            "info",
            f"Hybrid query ok — {result.iterations_used}/{result.max_iters} iter, "
            f"max error {result.final_max_error:.4g}, "
            f"converged: {'yes' if result.converged else 'no'}.",
        )

    @Slot(str, str)
    def _on_hybrid_failed(self, message: str, _tb: str) -> None:
        self.hybrid_failed.emit(message)
        self.log_message.emit("error", f"Hybrid query failed: {message}")
