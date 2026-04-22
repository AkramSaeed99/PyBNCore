"""NodeItem — read or author mode. Phase 2 adds ports and drag-to-connect."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from pybncore_gui.domain.node import NodeModel

NODE_WIDTH = 200.0
NODE_HEIGHT = 92.0
PORT_RADIUS = 8.0

_FILL = QColor("#f4f6fb")
_FILL_SELECTED = QColor("#dbe5ff")
_FILL_EVIDENCE = QColor("#fff4d6")
_BORDER = QColor("#3a4a66")
_BORDER_SELECTED = QColor("#1a56db")
_TEXT = QColor("#1b2333")
_SUBTEXT = QColor("#5a6478")
_PORT_IN = QColor("#6b7a98")
_PORT_OUT = QColor("#1a56db")
_PORT_HOVER = QColor("#e14c4c")


class NodeItem(QGraphicsObject):
    node_double_clicked = Signal(str)
    node_selected = Signal(str)

    def __init__(self, model: NodeModel, description: str = "") -> None:
        super().__init__()
        self._model = model
        self._description = description
        self._evidence_state: str | None = None
        self._hover_out_port = False
        self._hover_in_port = False
        self._drag_start_pos: QPointF | None = None
        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.setToolTip(self._build_tooltip())

    @property
    def model(self) -> NodeModel:
        return self._model

    @property
    def node_id(self) -> str:
        return self._model.id

    def set_evidence_state(self, state: str | None) -> None:
        if state == self._evidence_state:
            return
        self._evidence_state = state
        self.setToolTip(self._build_tooltip())
        self.update()

    def set_description(self, description: str) -> None:
        if description == self._description:
            return
        self._description = description
        self.setToolTip(self._build_tooltip())
        self.update()

    def update_model(self, model: NodeModel) -> None:
        self._model = model
        self.setToolTip(self._build_tooltip())
        self.update()

    def out_port_scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(NODE_WIDTH, NODE_HEIGHT / 2))

    def in_port_scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0.0, NODE_HEIGHT / 2))

    def out_port_rect(self) -> QRectF:
        return QRectF(
            NODE_WIDTH - PORT_RADIUS,
            NODE_HEIGHT / 2 - PORT_RADIUS,
            PORT_RADIUS * 2,
            PORT_RADIUS * 2,
        )

    def in_port_rect(self) -> QRectF:
        return QRectF(-PORT_RADIUS, NODE_HEIGHT / 2 - PORT_RADIUS, PORT_RADIUS * 2, PORT_RADIUS * 2)

    def boundingRect(self) -> QRectF:
        return QRectF(
            -PORT_RADIUS,
            -PORT_RADIUS,
            NODE_WIDTH + PORT_RADIUS * 2,
            NODE_HEIGHT + PORT_RADIUS * 2,
        )

    def shape(self):  # type: ignore[override]
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, NODE_WIDTH, NODE_HEIGHT), 10.0, 10.0)
        path.addEllipse(self.out_port_rect())
        path.addEllipse(self.in_port_rect())
        return path

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing)

        selected = self.isSelected()
        if self._evidence_state is not None:
            fill = _FILL_EVIDENCE
        elif selected:
            fill = _FILL_SELECTED
        else:
            fill = _FILL
        border = _BORDER_SELECTED if selected else _BORDER

        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(border, 1.8))
        painter.drawRoundedRect(QRectF(0, 0, NODE_WIDTH, NODE_HEIGHT), 10.0, 10.0)

        painter.setPen(QPen(_TEXT))
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRectF(14, 8, NODE_WIDTH - 28, 26),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._display_label(),
        )

        desc_font = QFont()
        desc_font.setPointSize(10)
        painter.setFont(desc_font)
        painter.setPen(QPen(_SUBTEXT))
        desc_rect = QRectF(14, 34, NODE_WIDTH - 28, 24)
        if self._description:
            painter.drawText(
                desc_rect,
                int(Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap),
                self._description,
            )

        sub_font = QFont()
        sub_font.setPointSize(10)
        painter.setFont(sub_font)
        painter.drawText(
            QRectF(14, NODE_HEIGHT - 40, NODE_WIDTH - 28, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._subtitle(),
        )

        if self._evidence_state is not None:
            ev_font = QFont()
            ev_font.setPointSize(11)
            ev_font.setBold(True)
            painter.setFont(ev_font)
            painter.setPen(QPen(QColor("#8a6100")))
            painter.drawText(
                QRectF(14, NODE_HEIGHT - 22, NODE_WIDTH - 28, 18),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"= {self._evidence_state}",
            )

        # Ports.
        painter.setPen(QPen(_BORDER, 1.0))
        painter.setBrush(QBrush(_PORT_HOVER if self._hover_in_port else _PORT_IN))
        painter.drawEllipse(self.in_port_rect())
        painter.setBrush(QBrush(_PORT_HOVER if self._hover_out_port else _PORT_OUT))
        painter.drawEllipse(self.out_port_rect())

    # --------------------------------------------------- hover / interactions

    def hoverMoveEvent(self, event) -> None:  # type: ignore[override]
        pos = event.pos()
        self._hover_out_port = self.out_port_rect().contains(pos)
        self._hover_in_port = self.in_port_rect().contains(pos)
        self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._hover_out_port = False
        self._hover_in_port = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        # Port / shift detection lives in the scene to avoid becoming the
        # mouse grabber and accidentally dragging the node along with the
        # pending edge.
        self._drag_start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.node_double_clicked.emit(self._model.id)
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemSelectedChange and bool(value):
            self.node_selected.emit(self._model.id)
        return super().itemChange(change, value)

    def _subtitle(self) -> str:
        parents = len(self._model.parents)
        states = len(self._model.states)
        if states:
            return f"{states} states · {parents} parent{'s' if parents != 1 else ''}"
        return f"{parents} parent{'s' if parents != 1 else ''}"

    def _display_label(self) -> str:
        return self._model.id

    def _build_tooltip(self) -> str:
        parts = [self._model.id]
        if self._description:
            parts.append(self._description)
        if self._model.states:
            parts.append("States: " + ", ".join(self._model.states))
        if self._model.parents:
            parts.append("Parents: " + ", ".join(self._model.parents))
        if self._evidence_state is not None:
            parts.append(f"Evidence: {self._evidence_state}")
        return "\n".join(parts)
