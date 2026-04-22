"""GraphScene — renders the portion of the model that lives in the currently
active sub-model.

Nodes whose `node_parent[id] == current_submodel_id` and sub-models whose
`parent_id == current_submodel_id` are shown. Edges are drawn only when both
endpoints are visible in the current scope. Cross-boundary edges are hidden in
Phase 1 of the sub-model feature.
"""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QGraphicsScene, QMenu

from pybncore_gui.domain.node import EdgeModel, NodeModel
from pybncore_gui.domain.submodel import ROOT_ID, SubModelLayout
from pybncore_gui.views.graph_canvas.edge_item import EdgeItem
from pybncore_gui.views.graph_canvas.ghost_anchor import GhostAnchorItem
from pybncore_gui.views.graph_canvas.layout import layered_positions
from pybncore_gui.views.graph_canvas.node_item import NodeItem
from pybncore_gui.views.graph_canvas.pending_edge import PendingEdge
from pybncore_gui.views.graph_canvas.submodel_item import SubModelItem


class GraphScene(QGraphicsScene):
    node_selected = Signal(str)
    node_double_clicked = Signal(str)
    edge_requested = Signal(str, str)
    nodes_moved = Signal(dict, dict)
    submodel_moved = Signal(str, float, float)
    delete_requested = Signal(str)
    rename_requested = Signal(str)
    remove_edge_requested = Signal(str, str)
    add_node_requested_at = Signal(float, float)
    palette_drop = Signal(str, float, float)        # kind, scene x, scene y
    enter_submodel_requested = Signal(str)
    add_submodel_requested = Signal(str)           # parent submodel id
    rename_submodel_requested = Signal(str)        # submodel id
    delete_submodel_requested = Signal(str)        # submodel id
    reparent_node_requested = Signal(str, str)     # node id, target submodel id

    def __init__(self) -> None:
        super().__init__()
        self._nodes: dict[str, NodeItem] = {}
        self._edges: list[EdgeItem] = []
        self._submodels: dict[str, SubModelItem] = {}
        self._ghosts: list[GhostAnchorItem] = []
        self._pending: Optional[PendingEdge] = None
        self._pending_source: Optional[NodeItem] = None
        self._positions_before_drag: Optional[dict[str, tuple[float, float]]] = None
        self._layout: SubModelLayout = SubModelLayout()
        self._current_submodel: str = ROOT_ID
        self.selectionChanged.connect(self._on_selection_changed)

    # ---------------------------------------------------------------- model

    def set_view(
        self,
        nodes: Iterable[NodeModel],
        edges: Iterable[EdgeModel],
        layout: SubModelLayout,
        current_submodel: str = ROOT_ID,
        positions: Optional[dict[str, tuple[float, float]]] = None,
        descriptions: Optional[dict[str, str]] = None,
    ) -> None:
        nodes = list(nodes)
        edges = list(edges)
        self._layout = layout
        self._current_submodel = current_submodel
        positions = positions or {}
        descriptions = descriptions or {}

        existing_positions = {
            nid: (item.pos().x(), item.pos().y()) for nid, item in self._nodes.items()
        }
        merged_positions = {**existing_positions, **positions}

        self.clear_graph()

        visible_node_ids = {
            n.id for n in nodes if layout.node_parent.get(n.id, ROOT_ID) == current_submodel
        }
        visible_submodel_ids = [
            sid for sid, sm in layout.submodels.items() if sm.parent_id == current_submodel
        ]

        visible_nodes = [n for n in nodes if n.id in visible_node_ids]
        auto_positions = layered_positions(visible_nodes, edges)

        for sid in visible_submodel_ids:
            sm = layout.submodels[sid]
            item = SubModelItem(sm)
            left, top, *_ = sm.position
            item.setPos(QPointF(left, top))
            item.double_clicked.connect(self.enter_submodel_requested.emit)
            self.addItem(item)
            self._submodels[sid] = item

        for node in visible_nodes:
            item = NodeItem(node, description=descriptions.get(node.id, ""))
            x, y = merged_positions.get(node.id, auto_positions.get(node.id, (0.0, 0.0)))
            item.setPos(QPointF(x, y))
            item.node_double_clicked.connect(self.node_double_clicked.emit)
            self.addItem(item)
            self._nodes[node.id] = item

        visible_submodel_set = set(visible_submodel_ids)

        def resolve_anchor(node_id: str):
            """Find the visible item representing `node_id` in the current scope.

            Returns a tuple (QGraphicsObject, is_submodel) or None if the node
            (and every ancestor sub-model) lives outside the current scope.
            """
            if node_id in self._nodes:
                return self._nodes[node_id], False
            parent = layout.node_parent.get(node_id, ROOT_ID)
            cur = parent
            while cur != ROOT_ID:
                if cur in visible_submodel_set:
                    return self._submodels[cur], True
                sm = layout.submodels.get(cur)
                if sm is None:
                    break
                cur = sm.parent_id
            return None

        seen_stubs: set[tuple[int, int]] = set()
        ghost_cache: dict[tuple[str, bool], GhostAnchorItem] = {}

        for edge in edges:
            src_anchor = resolve_anchor(edge.parent)
            dst_anchor = resolve_anchor(edge.child)

            # If one side has no visible anchor at all, synthesise a ghost
            # pinned to the scene margin so the user still sees the
            # connection.
            src_ghost = src_anchor is None
            dst_ghost = dst_anchor is None

            if src_ghost and dst_ghost:
                # Both endpoints invisible — nothing useful to draw.
                continue

            src_item, src_is_submodel = (
                src_anchor if src_anchor is not None else (None, False)
            )
            dst_item, dst_is_submodel = (
                dst_anchor if dst_anchor is not None else (None, False)
            )

            if src_ghost:
                src_item = self._get_or_create_ghost(
                    ghost_cache, edge.parent, on_right=False, reference=dst_item
                )
                src_is_submodel = True
            if dst_ghost:
                dst_item = self._get_or_create_ghost(
                    ghost_cache, edge.child, on_right=True, reference=src_item
                )
                dst_is_submodel = True

            if src_item is None or dst_item is None or src_item is dst_item:
                continue

            is_stub = src_is_submodel or dst_is_submodel
            if is_stub:
                key = (id(src_item), id(dst_item))
                if key in seen_stubs:
                    continue
                seen_stubs.add(key)
            edge_item = EdgeItem(src_item, dst_item, stub=is_stub)
            self.addItem(edge_item)
            self._edges.append(edge_item)

        self.setSceneRect(self.itemsBoundingRect().adjusted(-120, -120, 120, 120))
        self._refresh_edges()

    def clear_graph(self) -> None:
        for edge in self._edges:
            self.removeItem(edge)
        for item in self._nodes.values():
            self.removeItem(item)
        for sub in self._submodels.values():
            self.removeItem(sub)
        for ghost in self._ghosts:
            self.removeItem(ghost)
        self._edges.clear()
        self._nodes.clear()
        self._submodels.clear()
        self._ghosts.clear()
        self._cancel_pending_edge()
        self.setSceneRect(-200, -120, 400, 240)

    def current_positions(self) -> dict[str, tuple[float, float]]:
        return {nid: (item.pos().x(), item.pos().y()) for nid, item in self._nodes.items()}

    def submodel_positions(self) -> dict[str, tuple[float, float, float, float]]:
        out: dict[str, tuple[float, float, float, float]] = {}
        for sid, item in self._submodels.items():
            rect = item.boundingRect()
            left = item.pos().x()
            top = item.pos().y()
            out[sid] = (left, top, left + rect.width(), top + rect.height())
        return out

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

    def _submodel_item_at(self, scene_pos: QPointF) -> Optional[SubModelItem]:
        for item in self.items(scene_pos):
            if isinstance(item, SubModelItem):
                return item
        return None

    def _get_or_create_ghost(
        self,
        cache: dict[tuple[str, bool], GhostAnchorItem],
        node_id: str,
        *,
        on_right: bool,
        reference,
    ) -> GhostAnchorItem:
        key = (node_id, on_right)
        if key in cache:
            return cache[key]
        ghost = GhostAnchorItem(node_id, on_right=on_right)
        ghost.setToolTip(f"{node_id} — lives in another sub-model")
        # Position relative to the reference item (on the visible side) or,
        # if both endpoints are ghosts somehow, to the current items bbox.
        rect = self.itemsBoundingRect()
        if reference is not None:
            ref_rect = reference.mapRectToScene(reference.boundingRect())
            y = ref_rect.center().y() - ghost.boundingRect().height() / 2
            if on_right:
                x = ref_rect.right() + 120.0
            else:
                x = ref_rect.left() - 120.0 - ghost.boundingRect().width()
        else:
            y = rect.center().y()
            x = rect.right() + 60 if on_right else rect.left() - 60 - ghost.boundingRect().width()
        ghost.setPos(QPointF(x, y))
        self.addItem(ghost)
        self._ghosts.append(ghost)
        cache[key] = ghost
        return ghost

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
                target = self._nearest_node_within(scene_pos, radius=40.0)
            source = self._pending_source
            self._cancel_pending_edge()
            if target is not None and target is not source:
                self.edge_requested.emit(source.node_id, target.node_id)
            event.accept()
            return

        super().mouseReleaseEvent(event)
        self._refresh_edges()

        # Emit sub-model position updates on release so the layout persists.
        for sid, item in self._submodels.items():
            self.submodel_moved.emit(sid, item.pos().x(), item.pos().y())

        before = self._positions_before_drag
        self._positions_before_drag = None
        if before is None:
            return
        after = self.current_positions()
        diffs_before = {k: before[k] for k in after if k in before and before[k] != after[k]}
        diffs_after = {k: after[k] for k in diffs_before}

        # Drag-node-onto-container to reparent: if any moved node's centre
        # lands inside a visible sub-model box, request a reparent instead
        # of committing the move.
        reparent: tuple[str, str] | None = None
        for nid, (x, y) in diffs_after.items():
            node_item = self._nodes.get(nid)
            if node_item is None:
                continue
            br = node_item.boundingRect()
            centre = node_item.pos() + QPointF(
                br.width() / 2, br.height() / 2
            )
            for sid, sm_item in self._submodels.items():
                rect = sm_item.mapRectToScene(sm_item.boundingRect())
                if rect.contains(centre):
                    reparent = (nid, sid)
                    break
            if reparent is not None:
                break

        if reparent is not None:
            nid, target_sid = reparent
            # Snap the node back to its pre-drag position; the view will
            # re-render after reparenting and hide it from the current scope.
            if nid in before:
                node_item = self._nodes.get(nid)
                if node_item is not None:
                    node_item.setPos(QPointF(*before[nid]))
            self.reparent_node_requested.emit(nid, target_sid)
            return

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
        sub = self._submodel_item_at(event.scenePos())
        if isinstance(sub, SubModelItem):
            sid = sub.submodel_id
            menu = QMenu()
            enter = QAction(f"Enter '{sid}'", menu)
            enter.triggered.connect(
                lambda _=False, s=sid: self.enter_submodel_requested.emit(s)
            )
            menu.addAction(enter)
            menu.addSeparator()
            add_sub = QAction("Add Sub-Model inside…", menu)
            add_sub.triggered.connect(
                lambda _=False, s=sid: self.add_submodel_requested.emit(s)
            )
            rename_sub = QAction("Rename sub-model…", menu)
            rename_sub.triggered.connect(
                lambda _=False, s=sid: self.rename_submodel_requested.emit(s)
            )
            delete_sub = QAction("Delete sub-model", menu)
            delete_sub.triggered.connect(
                lambda _=False, s=sid: self.delete_submodel_requested.emit(s)
            )
            menu.addAction(add_sub)
            menu.addAction(rename_sub)
            menu.addAction(delete_sub)
            menu.exec(event.screenPos())
            return

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
            menu.addSeparator()
            # Move-to-submodel submenu
            move_menu = menu.addMenu("Move to sub-model")
            root_action = QAction("Root", move_menu)
            root_action.triggered.connect(
                lambda _=False, n=node_id: self.reparent_node_requested.emit(n, "")
            )
            move_menu.addAction(root_action)
            for sid, sm in sorted(self._layout.submodels.items(), key=lambda kv: kv[1].name):
                act = QAction(sm.name, move_menu)
                act.triggered.connect(
                    lambda _=False, n=node_id, target=sid: self.reparent_node_requested.emit(n, target)
                )
                move_menu.addAction(act)
        else:
            add_n = QAction("Add Discrete Node here…", menu)
            pos = event.scenePos()
            add_n.triggered.connect(
                lambda _=False, x=pos.x(), y=pos.y(): self.add_node_requested_at.emit(x, y)
            )
            menu.addAction(add_n)
            add_sm = QAction("Add Sub-Model here…", menu)
            add_sm.triggered.connect(
                lambda _=False: self.add_submodel_requested.emit(self._current_submodel)
            )
            menu.addAction(add_sm)

        menu.exec(event.screenPos())

    def _select_programmatic(self, node_id: str) -> None:
        self.select_node(node_id)
        self.node_selected.emit(node_id)

    # -------------------------------------------------- palette drag/drop

    def dragEnterEvent(self, event):  # type: ignore[override]
        from pybncore_gui.views.panels.palette_panel import PALETTE_MIME
        if event.mimeData().hasFormat(PALETTE_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):  # type: ignore[override]
        from pybncore_gui.views.panels.palette_panel import PALETTE_MIME
        if event.mimeData().hasFormat(PALETTE_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):  # type: ignore[override]
        from pybncore_gui.views.panels.palette_panel import PALETTE_MIME
        md = event.mimeData()
        if not md.hasFormat(PALETTE_MIME):
            super().dropEvent(event)
            return
        kind = bytes(md.data(PALETTE_MIME)).decode("utf-8", errors="ignore") or "discrete"
        pos = event.scenePos()
        self.palette_drop.emit(kind, pos.x(), pos.y())
        event.acceptProposedAction()

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
