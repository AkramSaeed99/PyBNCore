"""Directed edge (parent → child) drawn as a bezier with a small arrowhead.

The path is stroked (no fill) so the interior of the cubic curve does not fill
as a closed region. Only the arrowhead polygon is filled.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem

from pybncore_gui.views.graph_canvas.node_item import NODE_HEIGHT, NODE_WIDTH, NodeItem

_COLOR = QColor("#4a5a76")
_COLOR_SELECTED = QColor("#1a56db")
_ARROW_SIZE = 10.0
_PEN_WIDTH = 1.6


class EdgeItem(QGraphicsItem):
    def __init__(self, source: NodeItem, target: NodeItem) -> None:
        super().__init__()
        self._source = source
        self._target = target
        self._path: QPainterPath = QPainterPath()
        self._arrow: QPolygonF = QPolygonF()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(-1.0)
        self.update_path()

    @property
    def source(self) -> NodeItem:
        return self._source

    @property
    def target(self) -> NodeItem:
        return self._target

    # -------------------------------------------------- geometry / hit-testing

    def boundingRect(self) -> QRectF:
        rect = self._path.boundingRect()
        if not self._arrow.isEmpty():
            rect = rect.united(self._arrow.boundingRect())
        pad = _PEN_WIDTH + _ARROW_SIZE
        return rect.adjusted(-pad, -pad, pad, pad)

    def shape(self) -> QPainterPath:  # type: ignore[override]
        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        stroke = stroker.createStroke(self._path)
        if not self._arrow.isEmpty():
            head = QPainterPath()
            head.addPolygon(self._arrow)
            head.closeSubpath()
            stroke.addPath(head)
        return stroke

    # -------------------------------------------------------------- painting

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        painter.setRenderHint(QPainter.Antialiasing)
        color = _COLOR_SELECTED if self.isSelected() else _COLOR

        pen = QPen(color, _PEN_WIDTH)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self._path)

        if not self._arrow.isEmpty():
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(self._arrow)

    # ------------------------------------------------------- path computation

    def update_path(self) -> None:
        p1 = self._source.pos() + QPointF(NODE_WIDTH, NODE_HEIGHT / 2)
        p2 = self._target.pos() + QPointF(0.0, NODE_HEIGHT / 2)

        dx = max(60.0, abs(p2.x() - p1.x()) / 2.0)
        c1 = QPointF(p1.x() + dx, p1.y())
        c2 = QPointF(p2.x() - dx, p2.y())

        self.prepareGeometryChange()

        path = QPainterPath(p1)
        path.cubicTo(c1, c2, p2)
        self._path = path

        # Arrowhead: isoceles triangle at p2, pointing along tangent (c2 → p2).
        line = QLineF(c2, p2)
        angle = math.atan2(-line.dy(), line.dx())
        arrow_p1 = p2 - QPointF(
            math.sin(angle + math.pi / 3) * _ARROW_SIZE,
            math.cos(angle + math.pi / 3) * _ARROW_SIZE,
        )
        arrow_p2 = p2 - QPointF(
            math.sin(angle + math.pi - math.pi / 3) * _ARROW_SIZE,
            math.cos(angle + math.pi - math.pi / 3) * _ARROW_SIZE,
        )
        self._arrow = QPolygonF([p2, arrow_p1, arrow_p2])

        self.update()
