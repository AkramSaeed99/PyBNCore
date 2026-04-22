"""Clickable breadcrumb showing the current sub-model path."""
from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class BreadcrumbBar(QWidget):
    navigate = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self.set_path([("", "Root")])

    def set_path(self, segments: Iterable[tuple[str, str]]) -> None:
        # Clear existing children (keep the trailing stretch).
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        segs = list(segments)
        for i, (sid, name) in enumerate(segs):
            button = QPushButton(name)
            button.setFlat(True)
            button.setCursor(button.cursor())
            button.setStyleSheet(
                "QPushButton { padding: 2px 6px; color: #1d4ed8; }"
                "QPushButton:hover { text-decoration: underline; }"
            )
            button.clicked.connect(lambda _=False, s=sid: self.navigate.emit(s))
            if i == len(segs) - 1:
                button.setStyleSheet(
                    "QPushButton { padding: 2px 6px; color: #1b2333; font-weight: 600; }"
                )
                button.setEnabled(False)
            self._layout.insertWidget(i * 2, button)
            if i < len(segs) - 1:
                sep = QLabel("›")
                sep.setStyleSheet("color: #8792a8;")
                self._layout.insertWidget(i * 2 + 1, sep)
