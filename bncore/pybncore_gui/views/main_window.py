"""Main application window — Phase 3: soft evidence, MAP, scenarios, batch, settings."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import CompileStats, JTStats
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.dialogs import (
    AddNodeDialog,
    ContinuousNodeDialog,
    EditStatesDialog,
    EngineSettingsDialog,
    EquationNodeDialog,
    NoisyMaxWizard,
    RenameNodeDialog,
    ThresholdDialog,
)
from pybncore_gui.domain.submodel import ROOT_ID
from pybncore_gui.views.graph_canvas.breadcrumb import BreadcrumbBar
from pybncore_gui.views.graph_canvas.scene import GraphScene
from pybncore_gui.views.graph_canvas.view import GraphView
from pybncore_gui.views.panels.batch_evidence_panel import BatchEvidencePanel
from pybncore_gui.views.panels.benchmark_panel import BenchmarkPanel
from pybncore_gui.views.panels.compare_panel import ComparePanel
from pybncore_gui.views.panels.continuous_evidence_panel import ContinuousEvidencePanel
from pybncore_gui.views.panels.continuous_posterior_panel import ContinuousPosteriorPanel
from pybncore_gui.views.panels.evidence_panel import EvidencePanel
from pybncore_gui.views.panels.log_panel import LogPanel
from pybncore_gui.views.panels.map_panel import MAPPanel
from pybncore_gui.views.panels.model_explorer import ModelExplorerPanel
from pybncore_gui.views.panels.monte_carlo_panel import MonteCarloPanel
from pybncore_gui.views.panels.node_inspector import NodeInspectorPanel
from pybncore_gui.views.panels.palette_panel import PalettePanel
from pybncore_gui.views.panels.results_panel import ResultsPanel
from pybncore_gui.views.panels.scenarios_panel import ScenariosPanel
from pybncore_gui.views.panels.sensitivity_panel import SensitivityPanel
from pybncore_gui.views.panels.soft_evidence_panel import SoftEvidencePanel
from pybncore_gui.views.panels.validation_panel import ValidationPanel
from pybncore_gui.views.panels.voi_panel import VOIPanel


class MainWindow(QMainWindow):
    def __init__(self, viewmodel: MainViewModel) -> None:
        super().__init__()
        self._viewmodel = viewmodel
        self._last_dir: str = str(Path.cwd())
        self._last_jt_stats: JTStats | None = None

        self.setWindowTitle("PyBNCore GUI")
        self.resize(1500, 960)

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menu_and_toolbar()
        self._build_status_bar()
        self._bind_viewmodel()

    # ----------------------------------------------------------------- UI setup

    def _build_central(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._breadcrumb = BreadcrumbBar()
        self._breadcrumb.navigate.connect(self._on_breadcrumb_navigate)
        layout.addWidget(self._breadcrumb)
        self._scene = GraphScene()
        self._view = GraphView(self._scene)
        layout.addWidget(self._view, stretch=1)
        self.setCentralWidget(container)

    def _build_docks(self) -> None:
        self._palette = PalettePanel()
        self._explorer = ModelExplorerPanel(self._viewmodel)
        self._inspector = NodeInspectorPanel(self._viewmodel)
        self._evidence_panel = EvidencePanel(self._viewmodel)
        self._soft_evidence_panel = SoftEvidencePanel(self._viewmodel)
        self._results_panel = ResultsPanel(self._viewmodel)
        self._map_panel = MAPPanel(self._viewmodel)
        self._batch_panel = BatchEvidencePanel(self._viewmodel)
        self._compare_panel = ComparePanel(self._viewmodel)
        self._scenarios_panel = ScenariosPanel(self._viewmodel)
        self._log_panel = LogPanel(self._viewmodel)
        self._validation_panel = ValidationPanel(self._viewmodel)

        self._palette_dock = self._make_dock("Palette", self._palette, Qt.LeftDockWidgetArea)
        self._explorer_dock = self._make_dock("Model Explorer", self._explorer, Qt.LeftDockWidgetArea)
        self.splitDockWidget(self._palette_dock, self._explorer_dock, Qt.Vertical)

        self._inspector_dock = self._make_dock("Node Inspector", self._inspector, Qt.RightDockWidgetArea)

        inference_tabs = QTabWidget()
        inference_tabs.addTab(self._evidence_panel, "Evidence")
        inference_tabs.addTab(self._soft_evidence_panel, "Soft")
        inference_tabs.addTab(self._results_panel, "Results")
        inference_tabs.addTab(self._map_panel, "MAP")
        inference_tabs.addTab(self._compare_panel, "Compare")
        self._inference_dock = self._make_dock("Inference", inference_tabs, Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self._inspector_dock, self._inference_dock)
        self._inspector_dock.raise_()

        self._scenarios_dock = self._make_dock(
            "Scenarios", self._scenarios_panel, Qt.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._inference_dock, self._scenarios_dock)

        # --- Hybrid / continuous panels (Phase 5) ---
        self._continuous_evidence_panel = ContinuousEvidencePanel(self._viewmodel)
        self._continuous_posterior_panel = ContinuousPosteriorPanel(self._viewmodel)
        hybrid_tabs = QTabWidget()
        hybrid_tabs.addTab(self._continuous_evidence_panel, "Cont. Evidence")
        hybrid_tabs.addTab(self._continuous_posterior_panel, "Cont. Posterior")
        self._hybrid_dock = self._make_dock(
            "Hybrid", hybrid_tabs, Qt.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._inference_dock, self._hybrid_dock)

        # --- Analysis panels (Phase 4) ---
        self._sensitivity_panel = SensitivityPanel(self._viewmodel)
        self._voi_panel = VOIPanel(self._viewmodel)
        self._benchmark_panel = BenchmarkPanel(self._viewmodel)
        self._monte_carlo_panel = MonteCarloPanel(self._viewmodel)

        analysis_tabs = QTabWidget()
        analysis_tabs.addTab(self._sensitivity_panel, "Sensitivity")
        analysis_tabs.addTab(self._voi_panel, "VOI")
        analysis_tabs.addTab(self._monte_carlo_panel, "Monte Carlo")
        self._analysis_dock = self._make_dock(
            "Analysis", analysis_tabs, Qt.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._inference_dock, self._analysis_dock)

        self._batch_dock = self._make_dock("Batch", self._batch_panel, Qt.BottomDockWidgetArea)
        self._benchmark_dock = self._make_dock(
            "Benchmark", self._benchmark_panel, Qt.BottomDockWidgetArea
        )
        self._log_dock = self._make_dock("Log", self._log_panel, Qt.BottomDockWidgetArea)
        self._validation_dock = self._make_dock(
            "Validation", self._validation_panel, Qt.BottomDockWidgetArea
        )
        self.tabifyDockWidget(self._log_dock, self._validation_dock)
        self.tabifyDockWidget(self._log_dock, self._batch_dock)
        self.tabifyDockWidget(self._log_dock, self._benchmark_dock)
        self._log_dock.raise_()

    def _make_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(area, dock)
        return dock

    def _build_actions(self) -> None:
        # File
        self._action_new = QAction("&New Model", self)
        self._action_new.setShortcut(QKeySequence.New)
        self._action_new.triggered.connect(self._viewmodel.new_model)

        self._action_open = QAction("&Open XDSL…", self)
        self._action_open.setShortcut(QKeySequence.Open)
        self._action_open.triggered.connect(self._on_open_xdsl)

        self._action_import_bif = QAction("Import &BIF…", self)
        self._action_import_bif.triggered.connect(self._on_import_bif)

        self._action_save = QAction("&Save XDSL", self)
        self._action_save.setShortcut(QKeySequence.Save)
        self._action_save.triggered.connect(self._on_save)

        self._action_save_as = QAction("Save XDSL &As…", self)
        self._action_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._action_save_as.triggered.connect(self._on_save_as)

        self._action_save_project = QAction("Save &Project…", self)
        self._action_save_project.setShortcut(QKeySequence("Ctrl+Alt+S"))
        self._action_save_project.triggered.connect(self._on_save_project)

        self._action_open_project = QAction("Open Pro&ject…", self)
        self._action_open_project.setShortcut(QKeySequence("Ctrl+Alt+O"))
        self._action_open_project.triggered.connect(self._on_load_project)

        self._action_quit = QAction("&Quit", self)
        self._action_quit.setShortcut(QKeySequence.Quit)
        self._action_quit.triggered.connect(self.close)

        # Edit
        self._action_undo = self._viewmodel.undo_stack.createUndoAction(self, "&Undo")
        self._action_undo.setShortcut(QKeySequence.Undo)
        self._action_redo = self._viewmodel.undo_stack.createRedoAction(self, "&Redo")
        self._action_redo.setShortcut(QKeySequence.Redo)

        # Model
        self._action_add_node = QAction("&Add Discrete Node…", self)
        self._action_add_node.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self._action_add_node.triggered.connect(lambda: self._open_add_node_dialog())

        self._action_add_noisy_max = QAction("Add &Noisy-MAX Node…", self)
        self._action_add_noisy_max.setShortcut(QKeySequence("Ctrl+Shift+M"))
        self._action_add_noisy_max.triggered.connect(self._open_noisy_max_wizard)

        self._action_add_equation = QAction("Add &Equation Node…", self)
        self._action_add_equation.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self._action_add_equation.triggered.connect(self._open_equation_dialog)

        self._action_add_continuous = QAction("Add &Continuous Node…", self)
        self._action_add_continuous.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self._action_add_continuous.triggered.connect(self._open_continuous_dialog)

        self._action_add_threshold = QAction("Add &Threshold…", self)
        self._action_add_threshold.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self._action_add_threshold.triggered.connect(self._open_threshold_dialog)

        self._action_run_hybrid = QAction("Run &Hybrid Query", self)
        self._action_run_hybrid.setShortcut(QKeySequence("Ctrl+H"))
        self._action_run_hybrid.triggered.connect(self._viewmodel.run_hybrid)
        self._action_run_hybrid.setEnabled(False)

        self._action_new_submodel = QAction("Add Sub-&Model…", self)
        self._action_new_submodel.setShortcut(QKeySequence("Ctrl+G"))
        self._action_new_submodel.triggered.connect(self._on_new_submodel)

        self._action_move_to_submodel = QAction("Move Selected to Sub-Model…", self)
        self._action_move_to_submodel.triggered.connect(self._on_move_selected_to_submodel)

        self._action_exit_submodel = QAction("Exit Sub-Model", self)
        self._action_exit_submodel.setShortcut(QKeySequence("Alt+Up"))
        self._action_exit_submodel.triggered.connect(self._viewmodel.exit_submodel)

        self._action_rename = QAction("&Rename Selected…", self)
        self._action_rename.setShortcut(QKeySequence("F2"))
        self._action_rename.triggered.connect(self._on_rename_selected)

        self._action_delete = QAction("&Delete Selected", self)
        self._action_delete.setShortcut(QKeySequence.Delete)
        self._action_delete.triggered.connect(self._on_delete_selected)

        self._action_validate = QAction("&Validate Model", self)
        self._action_validate.setShortcut(QKeySequence("F7"))
        self._action_validate.triggered.connect(self._viewmodel.validate)

        # Inference
        self._action_compile = QAction("&Compile", self)
        self._action_compile.setShortcut("F5")
        self._action_compile.triggered.connect(self._viewmodel.compile)

        self._action_query = QAction("Run &Query", self)
        self._action_query.setShortcut("Ctrl+Return")
        self._action_query.triggered.connect(lambda: self._viewmodel.run_query())
        self._action_query.setEnabled(False)

        self._action_map = QAction("Run &MAP / MPE", self)
        self._action_map.setShortcut(QKeySequence("Ctrl+M"))
        self._action_map.triggered.connect(self._viewmodel.run_map)

        self._action_clear_evidence = QAction("Clear &Evidence", self)
        self._action_clear_evidence.triggered.connect(self._viewmodel.clear_evidence)

        self._action_clear_soft_evidence = QAction("Clear &Soft Evidence", self)
        self._action_clear_soft_evidence.triggered.connect(self._viewmodel.clear_soft_evidence)

        # Scenarios
        self._action_save_scenario = QAction("Save &Current as Scenario…", self)
        self._action_save_scenario.setShortcut(QKeySequence("Ctrl+B"))
        self._action_save_scenario.triggered.connect(self._on_save_scenario)

        # Tools
        self._action_engine_settings = QAction("&Engine Settings…", self)
        self._action_engine_settings.setShortcut(QKeySequence("Ctrl+,"))
        self._action_engine_settings.triggered.connect(self._on_engine_settings)

        # View
        self._action_fit = QAction("&Fit to Contents", self)
        self._action_fit.setShortcut("Ctrl+0")
        self._action_fit.triggered.connect(self._view.fit_to_contents)

        self._action_reset_zoom = QAction("&Reset Zoom", self)
        self._action_reset_zoom.triggered.connect(self._view.reset_zoom)

    def _build_menu_and_toolbar(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(self._action_new)
        file_menu.addAction(self._action_open)
        file_menu.addAction(self._action_import_bif)
        file_menu.addSeparator()
        file_menu.addAction(self._action_save)
        file_menu.addAction(self._action_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self._action_open_project)
        file_menu.addAction(self._action_save_project)
        file_menu.addSeparator()
        file_menu.addAction(self._action_quit)

        edit_menu = menu.addMenu("&Edit")
        edit_menu.addAction(self._action_undo)
        edit_menu.addAction(self._action_redo)

        model_menu = menu.addMenu("&Model")
        model_menu.addAction(self._action_add_node)
        model_menu.addAction(self._action_add_noisy_max)
        model_menu.addAction(self._action_add_equation)
        model_menu.addAction(self._action_add_continuous)
        model_menu.addAction(self._action_add_threshold)
        model_menu.addSeparator()
        model_menu.addAction(self._action_new_submodel)
        model_menu.addAction(self._action_move_to_submodel)
        model_menu.addAction(self._action_exit_submodel)
        model_menu.addSeparator()
        model_menu.addAction(self._action_rename)
        model_menu.addAction(self._action_delete)
        model_menu.addSeparator()
        model_menu.addAction(self._action_validate)

        analysis_menu = menu.addMenu("&Analysis")
        self._action_show_sensitivity = QAction("&Sensitivity Analysis", self)
        self._action_show_sensitivity.triggered.connect(self._analysis_dock.raise_)
        self._action_show_voi = QAction("&Value of Information", self)
        self._action_show_voi.triggered.connect(self._analysis_dock.raise_)
        self._action_show_benchmark = QAction("&Benchmark", self)
        self._action_show_benchmark.triggered.connect(self._benchmark_dock.raise_)
        self._action_show_monte_carlo = QAction("&Monte Carlo", self)
        self._action_show_monte_carlo.triggered.connect(self._analysis_dock.raise_)
        analysis_menu.addAction(self._action_show_sensitivity)
        analysis_menu.addAction(self._action_show_voi)
        analysis_menu.addAction(self._action_show_monte_carlo)
        analysis_menu.addAction(self._action_show_benchmark)

        inf_menu = menu.addMenu("&Inference")
        inf_menu.addAction(self._action_compile)
        inf_menu.addAction(self._action_query)
        inf_menu.addAction(self._action_map)
        inf_menu.addAction(self._action_run_hybrid)
        inf_menu.addSeparator()
        inf_menu.addAction(self._action_clear_evidence)
        inf_menu.addAction(self._action_clear_soft_evidence)

        scenarios_menu = menu.addMenu("&Scenarios")
        scenarios_menu.addAction(self._action_save_scenario)

        tools_menu = menu.addMenu("&Tools")
        tools_menu.addAction(self._action_engine_settings)

        view_menu = menu.addMenu("&View")
        view_menu.addAction(self._action_fit)
        view_menu.addAction(self._action_reset_zoom)

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self._action_new)
        toolbar.addAction(self._action_open)
        toolbar.addAction(self._action_save)
        toolbar.addSeparator()
        toolbar.addAction(self._action_undo)
        toolbar.addAction(self._action_redo)
        toolbar.addSeparator()
        toolbar.addAction(self._action_add_node)
        toolbar.addAction(self._action_add_noisy_max)
        toolbar.addAction(self._action_add_equation)
        toolbar.addAction(self._action_validate)
        toolbar.addSeparator()
        toolbar.addAction(self._action_compile)
        toolbar.addAction(self._action_query)
        toolbar.addAction(self._action_map)
        toolbar.addAction(self._action_clear_evidence)
        toolbar.addSeparator()
        toolbar.addAction(self._action_engine_settings)
        toolbar.addSeparator()
        toolbar.addAction(self._action_fit)

    def _build_status_bar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self._status_label = QLabel("Ready.")
        self._model_label = QLabel("")
        self._engine_label = QLabel("exact")
        self._engine_label.setStyleSheet("color: #4a5363;")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(160)
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        bar.addWidget(self._status_label, stretch=1)
        bar.addPermanentWidget(self._engine_label)
        bar.addPermanentWidget(self._model_label)
        bar.addPermanentWidget(self._progress)

    # ------------------------------------------------------------- bindings

    def _bind_viewmodel(self) -> None:
        vm = self._viewmodel
        vm.model_loaded.connect(self._on_model_loaded)
        vm.structure_changed.connect(self._on_structure_changed)
        vm.positions_changed.connect(self._on_positions_changed)
        vm.selection_changed.connect(self._on_selection_changed)
        vm.evidence_changed.connect(self._scene.apply_evidence)
        vm.busy_changed.connect(self._on_busy_changed)
        vm.compile_finished.connect(self._on_compile_finished)
        vm.compile_failed.connect(self._on_compile_failed)
        vm.query_failed.connect(self._on_query_failed)
        vm.jt_stats_updated.connect(self._on_jt_stats)
        vm.settings_changed.connect(self._on_settings_changed)

        self._scene.node_selected.connect(vm.select_node)
        self._scene.edge_requested.connect(vm.add_edge)
        self._scene.remove_edge_requested.connect(vm.remove_edge)
        self._scene.delete_requested.connect(self._on_scene_delete)
        self._scene.rename_requested.connect(self._on_scene_rename)
        self._scene.add_node_requested_at.connect(self._on_scene_add_node)
        self._scene.nodes_moved.connect(self._on_scene_nodes_moved)
        self._scene.enter_submodel_requested.connect(vm.enter_submodel)
        self._scene.submodel_moved.connect(vm.update_submodel_position)
        self._scene.palette_drop.connect(self._on_palette_drop)
        self._scene.add_submodel_requested.connect(self._on_scene_add_submodel)
        self._scene.rename_submodel_requested.connect(self._on_scene_rename_submodel)
        self._scene.delete_submodel_requested.connect(self._on_scene_delete_submodel)
        self._scene.reparent_node_requested.connect(self._on_scene_reparent_node)

        vm.layout_changed.connect(self._on_layout_changed)
        vm.current_submodel_changed.connect(self._on_current_submodel_changed)
        vm.continuous_nodes_changed.connect(self._on_continuous_nodes_changed)

        self._palette.add_discrete_node_requested.connect(lambda: self._open_add_node_dialog())
        self._inspector.rename_requested.connect(self._on_scene_rename)
        self._inspector.delete_requested.connect(self._on_scene_delete)
        self._inspector.edit_states_requested.connect(self._on_edit_states_requested)

    # ---------------------------------------------------------- dialogs / IO

    def _on_open_xdsl(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open XDSL",
            self._last_dir,
            "XDSL files (*.xdsl *.xml);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        self._viewmodel.open_xdsl(path)

    def _on_import_bif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import BIF",
            self._last_dir,
            "BIF files (*.bif);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        self._viewmodel.import_bif(path)

    def _on_save(self) -> None:
        current = self._viewmodel.session.source_path
        if not current:
            self._on_save_as()
            return
        self._viewmodel.save_xdsl(current)

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save XDSL",
            self._last_dir,
            "XDSL files (*.xdsl);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".xdsl"):
            path += ".xdsl"
        self._last_dir = str(Path(path).parent)
        self._viewmodel.save_xdsl(path)

    def _on_save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            self._last_dir,
            "PyBNCore project (*.pbnproj);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        self._viewmodel.save_project(path)

    def _on_load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._last_dir,
            "PyBNCore project (*.pbnproj);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        self._viewmodel.load_project(path)

    def _open_add_node_dialog(self, position: tuple[float, float] | None = None) -> None:
        existing = {n.id for n in self._viewmodel.model_service.list_nodes()}
        if not self._viewmodel.session.has_model:
            self._viewmodel.new_model()
            existing = set()
        dlg = AddNodeDialog(existing, parent=self)
        if not dlg.exec():
            return
        name, states = dlg.result_data()
        self._viewmodel.add_discrete_node(name, list(states))
        if position is not None:
            self._viewmodel.update_position(name, *position)

    def _on_scene_delete(self, node_id: str) -> None:
        if not node_id:
            return
        reply = QMessageBox.question(
            self,
            "Delete node",
            f"Delete node '{node_id}'? This will also remove incident edges.",
        )
        if reply == QMessageBox.Yes:
            self._viewmodel.remove_node(node_id)

    def _on_scene_rename(self, node_id: str) -> None:
        if not node_id:
            return
        existing = {n.id for n in self._viewmodel.model_service.list_nodes()}
        dlg = RenameNodeDialog(node_id, existing, parent=self)
        if not dlg.exec():
            return
        new_name = dlg.new_name()
        if new_name != node_id:
            self._viewmodel.rename_node(node_id, new_name)

    def _on_scene_add_node(self, x: float, y: float) -> None:
        self._open_add_node_dialog(position=(x, y))

    def _on_scene_nodes_moved(self, before: dict, after: dict) -> None:
        self._viewmodel.record_move(before, after)
        for nid, pos in after.items():
            self._viewmodel.update_position(nid, pos[0], pos[1])

    def _on_rename_selected(self) -> None:
        nid = self._viewmodel.selected_node
        if nid:
            self._on_scene_rename(nid)

    def _on_delete_selected(self) -> None:
        nid = self._viewmodel.selected_node
        if nid:
            self._on_scene_delete(nid)

    def _on_save_scenario(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save scenario", "Scenario name:")
        if not ok or not name.strip():
            return
        self._viewmodel.save_current_scenario(name.strip())

    def _open_noisy_max_wizard(self) -> None:
        if not self._viewmodel.session.has_model:
            self._viewmodel.new_model()
        existing = [n.id for n in self._viewmodel.model_service.list_nodes()]
        dlg = NoisyMaxWizard(existing, self._viewmodel.model_service, parent=self)
        if not dlg.exec():
            return
        payload = dlg.result_data()
        if payload is None:
            return
        name, states, parents, link_matrices, leak = payload
        self._viewmodel.add_noisy_max_node(name, list(states), list(parents), link_matrices, leak)

    def _open_equation_dialog(self) -> None:
        if not self._viewmodel.session.has_model:
            self._viewmodel.new_model()
        existing = [n.id for n in self._viewmodel.model_service.list_nodes()]
        dlg = EquationNodeDialog(existing, self._viewmodel.model_service, parent=self)
        if not dlg.exec():
            return
        payload = dlg.result_data()
        if payload is None:
            return
        name, states, parents, fn, source = payload
        self._viewmodel.add_equation_node(
            name, list(states), list(parents), fn, source
        )

    def _open_continuous_dialog(self) -> None:
        if not self._viewmodel.session.has_model:
            self._viewmodel.new_model()
        existing_ids = {n.id for n in self._viewmodel.model_service.list_nodes()}
        existing_ids.update(self._viewmodel.continuous_nodes)
        parents = sorted(existing_ids)
        dlg = ContinuousNodeDialog(existing_ids, parents, parent_widget=self)
        if not dlg.exec():
            return
        spec = dlg.result_spec()
        if spec is not None:
            self._viewmodel.add_continuous_node(spec)

    def _open_threshold_dialog(self) -> None:
        dlg = ThresholdDialog(list(self._viewmodel.continuous_nodes), parent=self)
        if not dlg.exec():
            return
        payload = dlg.result_data()
        if payload is None:
            return
        node, threshold = payload
        self._viewmodel.add_threshold(node, threshold)

    def _on_continuous_nodes_changed(self, names: list[str]) -> None:
        self._action_run_hybrid.setEnabled(
            bool(names) and not self._viewmodel.is_busy
        )

    def _on_engine_settings(self) -> None:
        # Try to refresh JT stats before showing the dialog — best-effort.
        if self._viewmodel.session.has_model:
            self._viewmodel.fetch_jt_stats()
        dlg = EngineSettingsDialog(
            settings=self._viewmodel.settings,
            jt_stats=self._last_jt_stats,
            parent=self,
        )
        if not dlg.exec():
            return
        updated = dlg.result_settings()
        if updated is not None:
            self._viewmodel.update_settings(updated)

    # -------------------------------------------------------- viewmodel hooks

    def _on_model_loaded(self, nodes, edges) -> None:
        self._scene.set_view(
            nodes,
            edges,
            self._viewmodel.layout,
            self._viewmodel.current_submodel,
            positions=self._viewmodel.positions,
            descriptions=self._viewmodel.descriptions,
        )
        self._update_breadcrumb()
        self._view.fit_to_contents()
        source = self._viewmodel.session.source_path or "untitled"
        self._model_label.setText(
            f"{Path(source).name}  —  {len(nodes)} nodes / {len(edges)} edges"
        )
        self._action_query.setEnabled(bool(self._viewmodel.selected_node))
        self._status_label.setText("Model loaded.")
        for nid, pos in self._scene.current_positions().items():
            self._viewmodel.update_position(nid, pos[0], pos[1])

    def _on_structure_changed(self, nodes, edges) -> None:
        self._scene.set_view(
            nodes,
            edges,
            self._viewmodel.layout,
            self._viewmodel.current_submodel,
            positions=self._viewmodel.positions,
            descriptions=self._viewmodel.descriptions,
        )
        self._update_breadcrumb()
        source = self._viewmodel.session.source_path or "untitled"
        self._model_label.setText(
            f"{Path(source).name}  —  {len(nodes)} nodes / {len(edges)} edges"
        )

    def _on_layout_changed(self, layout) -> None:
        self._update_breadcrumb()

    def _on_current_submodel_changed(self, submodel_id: str) -> None:
        self._update_breadcrumb()

    def _update_breadcrumb(self) -> None:
        self._breadcrumb.set_path(
            self._viewmodel.layout.breadcrumb(self._viewmodel.current_submodel)
        )
        self._action_exit_submodel.setEnabled(
            self._viewmodel.current_submodel != ROOT_ID
        )

    def _on_breadcrumb_navigate(self, submodel_id: str) -> None:
        self._viewmodel.enter_submodel(submodel_id)

    def _on_new_submodel(self) -> None:
        self._on_scene_add_submodel(self._viewmodel.current_submodel)

    def _on_palette_drop(self, kind: str, x: float, y: float) -> None:
        if kind == "discrete":
            self._open_add_node_dialog(position=(x, y))

    def _on_scene_add_submodel(self, parent_id: str) -> None:
        name, ok = QInputDialog.getText(self, "New sub-model", "Sub-model name:")
        if not ok or not name.strip():
            return
        self._viewmodel.create_submodel(name.strip(), parent_id)

    def _on_scene_rename_submodel(self, submodel_id: str) -> None:
        sm = self._viewmodel.layout.submodels.get(submodel_id)
        if sm is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename sub-model", "New name:", text=sm.name
        )
        if not ok or not new_name.strip() or new_name == sm.name:
            return
        self._viewmodel.rename_submodel(submodel_id, new_name.strip())

    def _on_scene_delete_submodel(self, submodel_id: str) -> None:
        sm = self._viewmodel.layout.submodels.get(submodel_id)
        if sm is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete sub-model",
            f"Delete sub-model '{sm.name}'? "
            "Its contents are re-homed to the parent.",
        )
        if reply != QMessageBox.Yes:
            return
        self._viewmodel.delete_submodel(submodel_id)

    def _on_scene_reparent_node(self, node_id: str, target_submodel: str) -> None:
        if not node_id:
            return
        self._viewmodel.move_nodes_to_submodel([node_id], target_submodel)

    def _on_edit_states_requested(self, node_id: str) -> None:
        if not node_id:
            return
        nodes = self._viewmodel.model_service.list_nodes()
        match = next((n for n in nodes if n.id == node_id), None)
        if match is None:
            return
        # Compute affected children so the dialog can warn the user.
        try:
            children = tuple(
                self._viewmodel.authoring_service.node_snapshot(node_id).children
            )
        except Exception:
            children = ()
        dlg = EditStatesDialog(node_id, list(match.states), children, parent=self)
        if not dlg.exec():
            return
        states = dlg.result_states()
        if states is None or states == match.states:
            return
        self._viewmodel.edit_node_states(node_id, list(states))

    def _on_move_selected_to_submodel(self) -> None:
        nid = self._viewmodel.selected_node
        if not nid:
            self._viewmodel.log_message.emit(
                "warning", "Select a node before moving it to a sub-model."
            )
            return
        layout = self._viewmodel.layout
        if not layout.submodels:
            QMessageBox.information(
                self,
                "No sub-models",
                "Create a sub-model first (Model → Add Sub-Model…).",
            )
            return
        names = [(ROOT_ID, "Root")] + [(sid, sm.name) for sid, sm in layout.submodels.items()]
        labels = [n for _, n in names]
        choice, ok = QInputDialog.getItem(
            self, "Move to sub-model", "Target:", labels, 0, False
        )
        if not ok:
            return
        target_id = names[labels.index(choice)][0]
        self._viewmodel.move_nodes_to_submodel([nid], target_id)

    def _on_positions_changed(self, positions: dict) -> None:
        self._scene.apply_positions(positions)

    def _on_selection_changed(self, node_id: str) -> None:
        self._scene.select_node(node_id)
        self._action_query.setEnabled(bool(node_id) and not self._viewmodel.is_busy)
        self._action_rename.setEnabled(bool(node_id))
        self._action_delete.setEnabled(bool(node_id))
        if node_id:
            self._status_label.setText(f"Selected: {node_id}")

    def _on_busy_changed(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self._action_compile.setEnabled(not busy)
        self._action_open.setEnabled(not busy)
        self._action_new.setEnabled(not busy)
        self._action_import_bif.setEnabled(not busy)
        self._action_save.setEnabled(not busy)
        self._action_save_as.setEnabled(not busy)
        self._action_save_project.setEnabled(not busy)
        self._action_open_project.setEnabled(not busy)
        self._action_map.setEnabled(not busy)
        self._action_query.setEnabled(not busy and bool(self._viewmodel.selected_node))
        self._action_run_hybrid.setEnabled(
            not busy and bool(self._viewmodel.continuous_nodes)
        )
        self._status_label.setText("Working…" if busy else "Ready.")

    def _on_compile_finished(self, stats: CompileStats) -> None:
        self._status_label.setText(
            f"Compiled: {stats.node_count} nodes, {stats.edge_count} edges."
        )

    def _on_compile_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Compile failed", message)

    def _on_query_failed(self, message: str) -> None:
        self._status_label.setText(f"Query failed: {message}")

    def _on_jt_stats(self, stats: JTStats) -> None:
        self._last_jt_stats = stats

    def _on_settings_changed(self, settings) -> None:
        label = "loopy-BP" if settings.use_loopy_bp else "exact"
        self._engine_label.setText(f"{label} · {settings.triangulation}")
