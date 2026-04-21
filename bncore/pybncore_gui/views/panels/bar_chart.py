"""Minimal painted bar chart. Phase 1 keeps dependencies to stock PySide6."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

_BAR_COLOR = QColor("#4178df")
_GRID_COLOR = QColor("#d0d4dc")
_LABEL_COLOR = QColor("#1b2333")
_AXIS_COLOR = QColor("#8792a8")


class BarChartWidget(QWidget):
    """Horizontal bar chart for probability distributions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._title: str = ""
        self._pairs: list[tuple[str, float]] = []

    def set_data(self, title: str, pairs: list[tuple[str, float]]) -> None:
        self._title = title
        self._pairs = list(pairs)
        self.update()

    def clear(self) -> None:
        self.set_data("", [])

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)

        if not self._pairs:
            painter.setPen(QPen(_AXIS_COLOR))
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No posterior to display.")
            return

        margin_left = 140
        margin_right = 48
        margin_top = 28 if self._title else 12
        margin_bottom = 20

        width = self.width()
        height = self.height()
        plot_w = max(40, width - margin_left - margin_right)
        plot_h = max(40, height - margin_top - margin_bottom)

        if self._title:
            painter.setPen(QPen(_LABEL_COLOR))
            title_font = QFont()
            title_font.setBold(True)
            title_font.setPointSize(10)
            painter.setFont(title_font)
            painter.drawText(
                QRectF(0, 4, width, 20),
                Qt.AlignCenter,
                self._title,
            )

        row_h = plot_h / max(1, len(self._pairs))
        max_value = max((v for _, v in self._pairs), default=1.0) or 1.0

        label_font = QFont()
        label_font.setPointSize(9)
        painter.setFont(label_font)

        for i, (label, value) in enumerate(self._pairs):
            y = margin_top + i * row_h
            bar_len = (value / max_value) * plot_w

            painter.setPen(QPen(_GRID_COLOR))
            painter.drawLine(margin_left, int(y + row_h), margin_left + plot_w, int(y + row_h))

            painter.setBrush(QBrush(_BAR_COLOR))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(
                QRectF(margin_left, y + row_h * 0.2, bar_len, row_h * 0.6),
                3.0,
                3.0,
            )

            painter.setPen(QPen(_LABEL_COLOR))
            painter.drawText(
                QRectF(4, y, margin_left - 10, row_h),
                Qt.AlignRight | Qt.AlignVCenter,
                label,
            )
            painter.drawText(
                QRectF(margin_left + bar_len + 6, y, margin_right + plot_w - bar_len, row_h),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"{value * 100:.2f} %" if max_value <= 1.0 else f"{value:.3f}",
            )

        painter.setPen(QPen(_AXIS_COLOR))
        painter.drawLine(
            margin_left, margin_top, margin_left, margin_top + plot_h
        )
