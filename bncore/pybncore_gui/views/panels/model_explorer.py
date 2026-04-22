"""Hierarchical tree: sub-models and the nodes they contain."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QTreeView, QVBoxLayout, QWidget

from pybncore_gui.domain.node import EdgeModel, NodeModel
from pybncore_gui.domain.submodel import ROOT_ID, ROOT_NAME, SubModelLayout
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_NODE_ID_ROLE = Qt.UserRole + 1
_SUBMODEL_ID_ROLE = Qt.UserRole + 2


class ModelExplorerPanel(QWidget):
    node_selected = Signal(str)

    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._nodes: list[NodeModel] = []
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._tree = QTreeView(self)
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setEditTriggers(QTreeView.NoEditTriggers)
        self._tree.setExpandsOnDoubleClick(True)
        self._model = QStandardItemModel(self)
        self._tree.setModel(self._model)
        self._tree.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._tree.doubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self._tree)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.model_loaded.connect(self._refresh)
        self._viewmodel.structure_changed.connect(self._refresh)
        self._viewmodel.layout_changed.connect(lambda *_: self._refresh(self._nodes, []))
        self._viewmodel.selection_changed.connect(self._reflect_selection)
        self._viewmodel.current_submodel_changed.connect(lambda *_: self._tree.viewport().update())

    def _refresh(self, nodes: list[NodeModel], edges: list[EdgeModel]) -> None:
        # Fall back to the live node list when called via layout_changed.
        if nodes:
            self._nodes = list(nodes)
        layout = self._viewmodel.layout
        self._model.clear()
        root = self._model.invisibleRootItem()

        root_item = QStandardItem(ROOT_NAME)
        root_item.setEditable(False)
        root_item.setData(ROOT_ID, _SUBMODEL_ID_ROLE)
        font = QFont()
        font.setBold(True)
        root_item.setFont(font)
        root.appendRow(root_item)
        self._populate(root_item, ROOT_ID, layout)
        self._tree.expand(root_item.index())

        # Expand down to the current sub-model so it's visible by default.
        current = self._viewmodel.current_submodel
        if current and current != ROOT_ID:
            self._expand_to(root_item, layout, current)

    def _populate(
        self,
        parent_item: QStandardItem,
        submodel_id: str,
        layout: SubModelLayout,
    ) -> None:
        # Sub-models first.
        for sid in sorted(layout.children_submodel_ids(submodel_id)):
            sm = layout.submodels[sid]
            item = QStandardItem(f"📁  {sm.name}")
            item.setEditable(False)
            item.setData(sid, _SUBMODEL_ID_ROLE)
            item.setToolTip(f"Sub-model {sid}")
            font = QFont()
            font.setBold(True)
            item.setFont(font)
            parent_item.appendRow(item)
            self._populate(item, sid, layout)

        # Then nodes that live directly at this level.
        alive = {n.id for n in self._nodes}
        direct_nodes = [
            n for n in self._nodes
            if layout.node_parent.get(n.id, ROOT_ID) == submodel_id and n.id in alive
        ]
        for node in sorted(direct_nodes, key=lambda n: n.id):
            label = f"{node.id}  [{len(node.states)}]"
            item = QStandardItem(label)
            item.setEditable(False)
            item.setData(node.id, _NODE_ID_ROLE)
            desc = self._viewmodel.descriptions.get(node.id, "")
            item.setToolTip(
                "\n".join(
                    filter(
                        None,
                        [
                            node.id,
                            desc,
                            "States: " + (", ".join(node.states) or "—"),
                            "Parents: " + (", ".join(node.parents) or "—"),
                        ],
                    )
                )
            )
            parent_item.appendRow(item)

    def _expand_to(
        self,
        root_item: QStandardItem,
        layout: SubModelLayout,
        target_submodel_id: str,
    ) -> None:
        path = layout.path_to(target_submodel_id)
        current = root_item
        self._tree.expand(current.index())
        for sid in path[1:]:
            found = None
            for row in range(current.rowCount()):
                child = current.child(row)
                if child and child.data(_SUBMODEL_ID_ROLE) == sid:
                    found = child
                    break
            if not found:
                return
            current = found
            self._tree.expand(current.index())

    def _on_selection_changed(self) -> None:
        indexes = self._tree.selectionModel().selectedIndexes()
        if not indexes:
            return
        node_id = indexes[0].data(_NODE_ID_ROLE)
        if node_id:
            self._viewmodel.select_node(node_id)

    def _on_double_clicked(self, index) -> None:
        sid = index.data(_SUBMODEL_ID_ROLE)
        if sid is not None:
            self._viewmodel.enter_submodel(sid)

    def _reflect_selection(self, node_id: str) -> None:
        if not node_id:
            self._tree.selectionModel().clearSelection()
            return

        def find(item: QStandardItem):
            for row in range(item.rowCount()):
                child = item.child(row)
                if child is None:
                    continue
                if child.data(_NODE_ID_ROLE) == node_id:
                    return child
                inner = find(child)
                if inner is not None:
                    return inner
            return None

        target = find(self._model.invisibleRootItem())
        if target is None:
            return
        self._tree.selectionModel().blockSignals(True)
        try:
            self._tree.setCurrentIndex(target.index())
            self._tree.scrollTo(target.index())
        finally:
            self._tree.selectionModel().blockSignals(False)
