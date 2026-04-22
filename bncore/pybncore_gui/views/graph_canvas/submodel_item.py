"""Graphical container for a sub-model. Double-click enters it."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from pybncore_gui.domain.submodel import SubModel

_TITLE_H = 34.0
_DEFAULT_W = 260.0
_DEFAULT_H = 150.0


class SubModelItem(QGraphicsObject):
    double_clicked = Signal(str)
    single_clicked = Signal(str)

    def __init__(self, model: SubModel) -> None:
        super().__init__()
        self._model = model
        self._fill = QColor(model.interior_color)
        self._border = QColor(model.outline_color)
        w = max(120.0, model.position[2] - model.position[0])
        h = max(80.0, model.position[3] - model.position[1])
        self._width = w if w > 0 else _DEFAULT_W
        self._height = h if h > 0 else _DEFAULT_H
        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(-2.0)
        self.setToolTip(
            f"Sub-model: {model.name}\nDouble-click to enter"
        )

    @property
    def submodel_id(self) -> str:
        return self._model.id

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # type: ignore[override]
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.boundingRect()

        # Shadow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 22)))
        painter.drawRoundedRect(rect.adjusted(3, 3, 3, 3), 14.0, 14.0)

        painter.setBrush(QBrush(self._fill))
        border_pen = QPen(self._border, 1.8 if not self.isSelected() else 2.4)
        border_pen.setStyle(Qt.SolidLine)
        painter.setPen(border_pen)
        painter.drawRoundedRect(rect, 14.0, 14.0)

        # Title bar
        title_rect = QRectF(0, 0, self._width, _TITLE_H)
        painter.setBrush(QBrush(self._border))
        painter.setPen(Qt.NoPen)
        title_path = QRectF(title_rect)
        painter.drawRoundedRect(title_path, 14.0, 14.0)
        painter.drawRect(QRectF(0, _TITLE_H - 14.0, self._width, 14.0))

        painter.setPen(QPen(QColor("#ffffff")))
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        painter.setFont(title_font)
        painter.drawText(
            title_rect.adjusted(38, 0, -12, 0),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._model.name,
        )

        # Folder glyph
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(10, 9, 20, 16), 3.0, 3.0)
        painter.drawRect(QRectF(10, 12, 9, 3))

        # Body hint
        sub_font = QFont()
        sub_font.setPointSize(11)
        sub_font.setItalic(True)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor("#334155")))
        painter.drawText(
            rect.adjusted(14, _TITLE_H + 6, -14, -8),
            Qt.AlignLeft | Qt.AlignTop,
            "Double-click to enter",
        )

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self.double_clicked.emit(self._model.id)
        super().mouseDoubleClickEvent(event)
