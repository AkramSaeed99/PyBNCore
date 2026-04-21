"""Rubber-banded edge shown while the user drags to connect two nodes."""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem


class PendingEdge(QGraphicsPathItem):
    def __init__(self, start: QPointF) -> None:
        super().__init__()
        pen = QPen(QColor("#4178df"), 1.8)
        pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        self.setZValue(10.0)
        self._start = start
        self._end = start
        self.update_path()

    def set_end(self, point: QPointF) -> None:
        self._end = point
        self.update_path()

    def update_path(self) -> None:
        path = QPainterPath(self._start)
        dx = max(40.0, abs(self._end.x() - self._start.x()) / 2.0)
        path.cubicTo(
            QPointF(self._start.x() + dx, self._start.y()),
            QPointF(self._end.x() - dx, self._end.y()),
            self._end,
        )
        self.setPath(path)
