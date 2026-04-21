"""Simple palette for Phase 2 — one button: Add Discrete Node."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class PalettePanel(QWidget):
    add_discrete_node_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Palette</b>"))
        self._add_btn = QPushButton("+  Add Discrete Node")
        self._add_btn.clicked.connect(self.add_discrete_node_requested.emit)
        layout.addWidget(self._add_btn)
        layout.addStretch()
