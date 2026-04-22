"""Marker shown at the scene edge for edges whose far endpoint is hidden.

When an edge connects a visible item to a node that lives in a sub-model
(and its sub-model) that has no visible container in the current scope,
we still want the user to see that the connection exists. A `GhostAnchor`
is a small pill rendered at the left or right margin showing the hidden
node's name; an `EdgeItem` with `stub=True` anchors to it.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

_FILL = QColor("#eef2ff")
_BORDER = QColor("#6b7a98")
_TEXT = QColor("#374151")
_PAD_X = 12.0
_HEIGHT = 26.0


class GhostAnchorItem(QGraphicsObject):
    def __init__(self, label: str, on_right: bool = False) -> None:
        super().__init__()
        self._label = str(label or "")
        self._on_right = bool(on_right)
        self._width = self._measure_width()
        self.setZValue(-1.5)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setToolTip(f"Hidden: {self._label}")

    def _measure_width(self) -> float:
        font = QFont()
        font.setPointSize(9)
        metrics = QFontMetrics(font)
        return float(metrics.horizontalAdvance(self._label) + _PAD_X * 2)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, _HEIGHT)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.boundingRect()
        painter.setBrush(QBrush(_FILL))
        painter.setPen(QPen(_BORDER, 1.2))
        painter.drawRoundedRect(rect, _HEIGHT / 2, _HEIGHT / 2)

        painter.setPen(QPen(_TEXT))
        font = QFont()
        font.setPointSize(9)
        font.setItalic(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, self._label)

    # Edge geometry uses left-center / right-center of the bounding box, so
    # these are adequate defaults — no port handling required.
