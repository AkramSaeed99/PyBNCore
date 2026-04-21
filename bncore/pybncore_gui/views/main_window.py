"""Main application window — Phase 2: authoring, undo/redo, save, import, validation."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from pybncore_gui.domain.results import CompileStats
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.dialogs import AddNodeDialog, RenameNodeDialog
from pybncore_gui.views.graph_canvas.scene import GraphScene
from pybncore_gui.views.graph_canvas.view import GraphView
from pybncore_gui.views.panels.evidence_panel import EvidencePanel
from pybncore_gui.views.panels.log_panel import LogPanel
from pybncore_gui.views.panels.model_explorer import ModelExplorerPanel
from pybncore_gui.views.panels.node_inspector import NodeInspectorPanel
from pybncore_gui.views.panels.palette_panel import PalettePanel
from pybncore_gui.views.panels.results_panel import ResultsPanel
from pybncore_gui.views.panels.validation_panel import ValidationPanel


class MainWindow(QMainWindow):
    def __init__(self, viewmodel: MainViewModel) -> None:
        super().__init__()
        self._viewmodel = viewmodel
        self._last_dir: str = str(Path.cwd())

        self.setWindowTitle("PyBNCore GUI")
        self.resize(1400, 900)

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menu_and_toolbar()
        self._build_status_bar()
        self._bind_viewmodel()

    # ----------------------------------------------------------------- UI setup

    def _build_central(self) -> None:
        self._scene = GraphScene()
        self._view = GraphView(self._scene)
        self.setCentralWidget(self._view)

    def _build_docks(self) -> None:
        self._palette = PalettePanel()
        self._explorer = ModelExplorerPanel(self._viewmodel)
        self._inspector = NodeInspectorPanel(self._viewmodel)
        self._evidence_panel = EvidencePanel(self._viewmodel)
        self._results_panel = ResultsPanel(self._viewmodel)
        self._log_panel = LogPanel(self._viewmodel)
        self._validation_panel = ValidationPanel(self._viewmodel)

        self._palette_dock = self._make_dock("Palette", self._palette, Qt.LeftDockWidgetArea)
        self._explorer_dock = self._make_dock("Model Explorer", self._explorer, Qt.LeftDockWidgetArea)
        self.splitDockWidget(self._palette_dock, self._explorer_dock, Qt.Vertical)

        self._inspector_dock = self._make_dock("Node Inspector", self._inspector, Qt.RightDockWidgetArea)

        right_tabs = QTabWidget()
        right_tabs.addTab(self._evidence_panel, "Evidence")
        right_tabs.addTab(self._results_panel, "Results")
        self._right_dock = self._make_dock("Inference", right_tabs, Qt.RightDockWidgetArea)
        self.tabifyDockWidget(self._inspector_dock, self._right_dock)
        self._inspector_dock.raise_()

        self._log_dock = self._make_dock("Log", self._log_panel, Qt.BottomDockWidgetArea)
        self._validation_dock = self._make_dock(
            "Validation", self._validation_panel, Qt.BottomDockWidgetArea
        )
        self.tabifyDockWidget(self._log_dock, self._validation_dock)
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

        self._action_clear_evidence = QAction("Clear &Evidence", self)
        self._action_clear_evidence.triggered.connect(self._viewmodel.clear_evidence)

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
        model_menu.addAction(self._action_rename)
        model_menu.addAction(self._action_delete)
        model_menu.addSeparator()
        model_menu.addAction(self._action_validate)

        inf_menu = menu.addMenu("&Inference")
        inf_menu.addAction(self._action_compile)
        inf_menu.addAction(self._action_query)
        inf_menu.addSeparator()
        inf_menu.addAction(self._action_clear_evidence)

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
        toolbar.addAction(self._action_validate)
        toolbar.addSeparator()
        toolbar.addAction(self._action_compile)
        toolbar.addAction(self._action_query)
        toolbar.addAction(self._action_clear_evidence)
        toolbar.addSeparator()
        toolbar.addAction(self._action_fit)

    def _build_status_bar(self) -> None:
        bar = QStatusBar(self)
        self.setStatusBar(bar)
        self._status_label = QLabel("Ready.")
        self._model_label = QLabel("")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(160)
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        bar.addWidget(self._status_label, stretch=1)
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

        # Canvas interactions
        self._scene.node_selected.connect(vm.select_node)
        self._scene.edge_requested.connect(vm.add_edge)
        self._scene.remove_edge_requested.connect(vm.remove_edge)
        self._scene.delete_requested.connect(self._on_scene_delete)
        self._scene.rename_requested.connect(self._on_scene_rename)
        self._scene.add_node_requested_at.connect(self._on_scene_add_node)
        self._scene.nodes_moved.connect(self._on_scene_nodes_moved)

        # Palette + inspector actions
        self._palette.add_discrete_node_requested.connect(lambda: self._open_add_node_dialog())
        self._inspector.rename_requested.connect(self._on_scene_rename)
        self._inspector.delete_requested.connect(self._on_scene_delete)

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
            # Auto-create an empty model so the user can start from a fresh canvas.
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
        # Convert tuples: scene emits dict[str, tuple[float, float]]
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

    # -------------------------------------------------------- viewmodel hooks

    def _on_model_loaded(self, nodes, edges) -> None:
        self._scene.set_model(nodes, edges, positions=self._viewmodel.positions)
        self._view.fit_to_contents()
        source = self._viewmodel.session.source_path or "untitled"
        self._model_label.setText(
            f"{Path(source).name}  —  {len(nodes)} nodes / {len(edges)} edges"
        )
        self._action_query.setEnabled(bool(self._viewmodel.selected_node))
        self._status_label.setText("Model loaded.")
        # Sync initial positions into the viewmodel so the sidecar can save them.
        for nid, pos in self._scene.current_positions().items():
            self._viewmodel.update_position(nid, pos[0], pos[1])

    def _on_structure_changed(self, nodes, edges) -> None:
        self._scene.set_model(nodes, edges, positions=self._viewmodel.positions)
        source = self._viewmodel.session.source_path or "untitled"
        self._model_label.setText(
            f"{Path(source).name}  —  {len(nodes)} nodes / {len(edges)} edges"
        )

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
        self._action_query.setEnabled(not busy and bool(self._viewmodel.selected_node))
        self._status_label.setText("Working…" if busy else "Ready.")

    def _on_compile_finished(self, stats: CompileStats) -> None:
        self._status_label.setText(
            f"Compiled: {stats.node_count} nodes, {stats.edge_count} edges."
        )

    def _on_compile_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Compile failed", message)

    def _on_query_failed(self, message: str) -> None:
        self._status_label.setText(f"Query failed: {message}")
