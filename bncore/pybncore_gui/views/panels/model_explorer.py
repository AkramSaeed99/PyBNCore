"""Tree view listing nodes and edges in the current model."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QTreeView, QVBoxLayout, QWidget

from pybncore_gui.domain.node import EdgeModel, NodeModel
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_NODE_ID_ROLE = Qt.UserRole + 1


class ModelExplorerPanel(QWidget):
    node_selected = Signal(str)

    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._tree = QTreeView(self)
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setEditTriggers(QTreeView.NoEditTriggers)
        self._model = QStandardItemModel(self)
        self._tree.setModel(self._model)
        self._tree.selectionModel().selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.model_loaded.connect(self._populate)
        self._viewmodel.selection_changed.connect(self._reflect_selection)

    def _populate(self, nodes: list[NodeModel], edges: list[EdgeModel]) -> None:
        self._model.clear()
        root = self._model.invisibleRootItem()

        nodes_group = QStandardItem(f"Nodes ({len(nodes)})")
        nodes_group.setEditable(False)
        for node in sorted(nodes, key=lambda n: n.id):
            item = QStandardItem(f"{node.id}  [{len(node.states)}]")
            item.setEditable(False)
            item.setData(node.id, _NODE_ID_ROLE)
            item.setToolTip(
                "\n".join(
                    [
                        f"States: {', '.join(node.states) or '—'}",
                        f"Parents: {', '.join(node.parents) or '—'}",
                    ]
                )
            )
            nodes_group.appendRow(item)
        root.appendRow(nodes_group)

        edges_group = QStandardItem(f"Edges ({len(edges)})")
        edges_group.setEditable(False)
        for edge in edges:
            item = QStandardItem(f"{edge.parent} → {edge.child}")
            item.setEditable(False)
            edges_group.appendRow(item)
        root.appendRow(edges_group)

        self._tree.expand(nodes_group.index())

    def _on_selection_changed(self) -> None:
        indexes = self._tree.selectionModel().selectedIndexes()
        if not indexes:
            return
        node_id = indexes[0].data(_NODE_ID_ROLE)
        if node_id:
            self._viewmodel.select_node(node_id)

    def _reflect_selection(self, node_id: str) -> None:
        if not node_id:
            self._tree.selectionModel().clearSelection()
            return
        # Find the matching QStandardItem and select it.
        for row in range(self._model.rowCount()):
            group = self._model.item(row)
            if group is None:
                continue
            for child_row in range(group.rowCount()):
                child = group.child(child_row)
                if child and child.data(_NODE_ID_ROLE) == node_id:
                    index = child.index()
                    self._tree.selectionModel().blockSignals(True)
                    try:
                        self._tree.setCurrentIndex(index)
                    finally:
                        self._tree.selectionModel().blockSignals(False)
                    return
