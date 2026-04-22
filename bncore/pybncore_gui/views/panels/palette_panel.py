"""Palette — click OR drag to create discrete nodes on the canvas."""
from __future__ import annotations

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag, QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

PALETTE_MIME = "application/x-pybncore-node-kind"


class _DraggableButton(QPushButton):
    def __init__(self, text: str, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._kind = kind
        self._drag_origin: QPoint | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is None or not (event.buttons() & Qt.LeftButton):
            return super().mouseMoveEvent(event)
        distance = (event.position().toPoint() - self._drag_origin).manhattanLength()
        if distance < 8:
            return super().mouseMoveEvent(event)

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(PALETTE_MIME, self._kind.encode("utf-8"))
        drag.setMimeData(mime)
        # Render the button as the drag pixmap.
        pix = QPixmap(self.size())
        self.render(pix)
        drag.setPixmap(pix)
        drag.setHotSpot(event.position().toPoint())
        drag.exec(Qt.CopyAction)
        self._drag_origin = None


class PalettePanel(QWidget):
    add_discrete_node_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Palette</b>"))
        layout.addWidget(QLabel("Click to add, or drag onto the canvas."))
        self._add_btn = _DraggableButton("+  Discrete Node", "discrete")
        self._add_btn.clicked.connect(self.add_discrete_node_requested.emit)
        layout.addWidget(self._add_btn)
        layout.addStretch()
