"""Directed edge. Endpoints may be nodes or sub-model containers."""
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
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

_COLOR = QColor("#2f3a52")
_COLOR_SELECTED = QColor("#1a56db")
_COLOR_STUB = QColor("#7080a0")
_ARROW_LENGTH = 16.0
_ARROW_HALF_WIDTH = 7.0
_PEN_WIDTH = 1.8
_TIP_MARGIN = 10.0


class EdgeItem(QGraphicsItem):
    def __init__(
        self,
        source: QGraphicsObject,
        target: QGraphicsObject,
        *,
        stub: bool = False,
    ) -> None:
        super().__init__()
        self._source = source
        self._target = target
        self._stub = bool(stub)
        self._path: QPainterPath = QPainterPath()
        self._arrow: QPolygonF = QPolygonF()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(-1.0)
        self.update_path()

    @property
    def source(self) -> QGraphicsObject:
        return self._source

    @property
    def target(self) -> QGraphicsObject:
        return self._target

    # -------------------------------------------------- geometry / hit-testing

    def boundingRect(self) -> QRectF:
        rect = self._path.boundingRect()
        if not self._arrow.isEmpty():
            rect = rect.united(self._arrow.boundingRect())
        pad = _PEN_WIDTH + _ARROW_LENGTH
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
        if self.isSelected():
            color = _COLOR_SELECTED
        elif self._stub:
            color = _COLOR_STUB
        else:
            color = _COLOR

        pen = QPen(color, _PEN_WIDTH)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        if self._stub:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self._path)

        if not self._arrow.isEmpty():
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(self._arrow)

    # ------------------------------------------------------- path computation

    @staticmethod
    def _anchor_points(item: QGraphicsObject, side: str) -> QPointF:
        rect = item.boundingRect()
        pos = item.pos()
        y = pos.y() + rect.top() + rect.height() / 2
        if side == "right":
            x = pos.x() + rect.right()
        else:
            x = pos.x() + rect.left()
        return QPointF(x, y)

    def update_path(self) -> None:
        source_anchor = self._anchor_points(self._source, "right")
        target_anchor = self._anchor_points(self._target, "left")

        # Tip sits just outside the target item.
        if target_anchor.x() >= source_anchor.x():
            tip = QPointF(target_anchor.x() - _TIP_MARGIN, target_anchor.y())
            control_x_src = source_anchor.x()
            control_x_tgt = tip.x()
        else:
            # Target is to the left of the source — route around with a dip.
            tip = QPointF(target_anchor.x() - _TIP_MARGIN, target_anchor.y())
            control_x_src = source_anchor.x() + 80
            control_x_tgt = tip.x() - 80

        dx = max(60.0, abs(tip.x() - source_anchor.x()) / 2.0)
        c1 = QPointF(control_x_src + dx, source_anchor.y())
        c2 = QPointF(control_x_tgt - dx, tip.y())

        self.prepareGeometryChange()

        line = QLineF(c2, tip)
        length = math.hypot(line.dx(), line.dy()) or 1.0
        dir_x = line.dx() / length
        dir_y = line.dy() / length
        nx, ny = -dir_y, dir_x

        base = QPointF(
            tip.x() - dir_x * _ARROW_LENGTH,
            tip.y() - dir_y * _ARROW_LENGTH,
        )
        left = QPointF(
            base.x() + nx * _ARROW_HALF_WIDTH,
            base.y() + ny * _ARROW_HALF_WIDTH,
        )
        right = QPointF(
            base.x() - nx * _ARROW_HALF_WIDTH,
            base.y() - ny * _ARROW_HALF_WIDTH,
        )

        path = QPainterPath(source_anchor)
        path.cubicTo(c1, c2, base)
        self._path = path
        self._arrow = QPolygonF([tip, left, right])

        self.update()
