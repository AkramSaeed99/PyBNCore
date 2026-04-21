"""GraphScene — reconciles NodeModel / EdgeModel lists with Qt items.

Phase 2 additions:
- drag-to-connect (click a node's output port, drag to another node)
- context menus (right-click on node or empty canvas)
- move coalescing (emits `nodes_moved` with before/after maps so the viewmodel
  can push a single MoveNodesCommand)
"""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QGraphicsScene, QMenu

from pybncore_gui.domain.node import EdgeModel, NodeModel
from pybncore_gui.views.graph_canvas.edge_item import EdgeItem
from pybncore_gui.views.graph_canvas.layout import layered_positions
from pybncore_gui.views.graph_canvas.node_item import NodeItem
from pybncore_gui.views.graph_canvas.pending_edge import PendingEdge


class GraphScene(QGraphicsScene):
    node_selected = Signal(str)
    node_double_clicked = Signal(str)
    edge_requested = Signal(str, str)
    nodes_moved = Signal(dict, dict)                  # before, after
    delete_requested = Signal(str)                    # node id
    rename_requested = Signal(str)
    remove_edge_requested = Signal(str, str)
    add_node_requested_at = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[str, NodeItem] = {}
        self._edges: list[EdgeItem] = []
        self._pending: Optional[PendingEdge] = None
        self._pending_source: Optional[NodeItem] = None
        self._positions_before_drag: Optional[dict[str, tuple[float, float]]] = None
        self.selectionChanged.connect(self._on_selection_changed)

    # ---------------------------------------------------------------- model

    def set_model(
        self,
        nodes: Iterable[NodeModel],
        edges: Iterable[EdgeModel],
        positions: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        nodes = list(nodes)
        edges = list(edges)
        if positions is None:
            positions = {}

        # Preserve existing positions where possible.
        existing_positions = {nid: (item.pos().x(), item.pos().y()) for nid, item in self._nodes.items()}
        merged_positions = {**existing_positions, **positions}

        self.clear_graph()
        auto_positions = layered_positions(nodes, edges)
        for node in nodes:
            item = NodeItem(node)
            x, y = merged_positions.get(node.id, auto_positions.get(node.id, (0.0, 0.0)))
            item.setPos(QPointF(x, y))
            item.node_double_clicked.connect(self.node_double_clicked.emit)
            self.addItem(item)
            self._nodes[node.id] = item

        for edge in edges:
            src = self._nodes.get(edge.parent)
            dst = self._nodes.get(edge.child)
            if src is None or dst is None:
                continue
            edge_item = EdgeItem(src, dst)
            self.addItem(edge_item)
            self._edges.append(edge_item)

        self.setSceneRect(self.itemsBoundingRect().adjusted(-120, -120, 120, 120))
        self._refresh_edges()

    def clear_graph(self) -> None:
        for edge in self._edges:
            self.removeItem(edge)
        for item in self._nodes.values():
            self.removeItem(item)
        self._edges.clear()
        self._nodes.clear()
        self._cancel_pending_edge()
        self.setSceneRect(-200, -120, 400, 240)

    def current_positions(self) -> dict[str, tuple[float, float]]:
        return {nid: (item.pos().x(), item.pos().y()) for nid, item in self._nodes.items()}

    def apply_positions(self, positions: dict[str, tuple[float, float]]) -> None:
        self.blockSignals(True)
        try:
            for nid, (x, y) in positions.items():
                item = self._nodes.get(nid)
                if item is not None:
                    item.setPos(QPointF(float(x), float(y)))
        finally:
            self.blockSignals(False)
        self._refresh_edges()

    def select_node(self, node_id: str) -> None:
        self.blockSignals(True)
        try:
            for nid, item in self._nodes.items():
                item.setSelected(nid == node_id)
        finally:
            self.blockSignals(False)

    def apply_evidence(self, evidence: dict[str, str]) -> None:
        for nid, item in self._nodes.items():
            item.set_evidence_state(evidence.get(nid))

    # ------------------------------------------------------ drag-to-connect

    def begin_drag_edge(self, source: NodeItem, start_scene: QPointF) -> None:
        self._cancel_pending_edge()
        self._pending_source = source
        self._pending = PendingEdge(source.out_port_scene_pos())
        self._pending.set_end(start_scene)
        self.addItem(self._pending)

    def _cancel_pending_edge(self) -> None:
        if self._pending is not None:
            self.removeItem(self._pending)
        self._pending = None
        self._pending_source = None

    def _node_item_at(self, scene_pos: QPointF) -> Optional[NodeItem]:
        for item in self.items(scene_pos):
            if isinstance(item, NodeItem):
                return item
        return None

    def _nearest_node_within(
        self, scene_pos: QPointF, radius: float
    ) -> Optional[NodeItem]:
        best: Optional[NodeItem] = None
        best_d2 = radius * radius
        for item in self._nodes.values():
            rect = item.mapRectToScene(item.boundingRect())
            dx = max(rect.left() - scene_pos.x(), 0.0, scene_pos.x() - rect.right())
            dy = max(rect.top() - scene_pos.y(), 0.0, scene_pos.y() - rect.bottom())
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = item
        return best

    # --------------------------------------------------------- mouse events

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            scene_pos = event.scenePos()
            node = self._node_item_at(scene_pos)
            if node is not None:
                local = node.mapFromScene(scene_pos)
                on_port = node.out_port_rect().contains(local)
                shift = bool(event.modifiers() & Qt.ShiftModifier)
                if on_port or shift:
                    self.begin_drag_edge(node, scene_pos)
                    event.accept()
                    return
            self._positions_before_drag = self.current_positions()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._pending is not None:
            self._pending.set_end(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)
        if event.buttons():
            self._refresh_edges()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._pending is not None and self._pending_source is not None:
            scene_pos = event.scenePos()
            target = self._node_item_at(scene_pos)
            if target is None:
                # Forgiving snap: if the cursor is near any node, take it.
                target = self._nearest_node_within(scene_pos, radius=40.0)
            source = self._pending_source
            self._cancel_pending_edge()
            if target is not None and target is not source:
                self.edge_requested.emit(source.node_id, target.node_id)
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self._refresh_edges()

        before = self._positions_before_drag
        self._positions_before_drag = None
        if before is None:
            return
        after = self.current_positions()
        diffs_before = {k: before[k] for k in after if k in before and before[k] != after[k]}
        diffs_after = {k: after[k] for k in diffs_before}
        if diffs_before:
            self.nodes_moved.emit(diffs_before, diffs_after)

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in list(self.selectedItems()):
                if isinstance(item, NodeItem):
                    self.delete_requested.emit(item.node_id)
                elif isinstance(item, EdgeItem):
                    self.remove_edge_requested.emit(item.source.node_id, item.target.node_id)
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._pending is not None:
            self._cancel_pending_edge()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):  # type: ignore[override]
        item = self._node_item_at(event.scenePos())
        menu = QMenu()
        if isinstance(item, NodeItem):
            node_id = item.node_id
            select = QAction(f"Select '{node_id}'", menu)
            select.triggered.connect(lambda _=False, n=node_id: self._select_programmatic(n))
            rename = QAction("Rename…", menu)
            rename.triggered.connect(lambda _=False, n=node_id: self.rename_requested.emit(n))
            delete = QAction("Delete", menu)
            delete.triggered.connect(lambda _=False, n=node_id: self.delete_requested.emit(n))
            menu.addAction(select)
            menu.addSeparator()
            menu.addAction(rename)
            menu.addAction(delete)
        else:
            add = QAction("Add Discrete Node here…", menu)
            pos = event.scenePos()
            add.triggered.connect(
                lambda _=False, x=pos.x(), y=pos.y(): self.add_node_requested_at.emit(x, y)
            )
            menu.addAction(add)

        menu.exec(event.screenPos())

    def _select_programmatic(self, node_id: str) -> None:
        self.select_node(node_id)
        self.node_selected.emit(node_id)

    # ------------------------------------------------------------- helpers

    def _refresh_edges(self) -> None:
        for edge in self._edges:
            edge.update_path()

    def _on_selection_changed(self) -> None:
        selected = self.selectedItems()
        if not selected:
            self.node_selected.emit("")
            return
        for item in selected:
            if isinstance(item, NodeItem):
                self.node_selected.emit(item.node_id)
                return
