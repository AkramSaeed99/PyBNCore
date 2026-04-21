"""Bottom log panel — receives `(level, message)` from the viewmodel."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_COLORS = {
    "info": QColor("#1b2333"),
    "warning": QColor("#9b6a00"),
    "error": QColor("#8a1c1c"),
    "debug": QColor("#6a7387"),
}


class LogPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._view = QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(2000)
        layout.addWidget(self._view)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.log_message.connect(self._append)

    def _append(self, level: str, message: str) -> None:
        color = _COLORS.get(level, _COLORS["info"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"[{timestamp}] {level.upper():7s} ", fmt)
        plain = QTextCharFormat()
        plain.setForeground(_COLORS["info"])
        cursor.insertText(message + "\n", plain)
        self._view.setTextCursor(cursor)
        self._view.ensureCursorVisible()
