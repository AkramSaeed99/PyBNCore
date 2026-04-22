"""Continuous posterior viewer — PDF/CDF plot, quantiles, tail probabilities."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import ContinuousPosteriorDTO, HybridResultDTO
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_PLOT_COLOR = QColor("#1d4ed8")
_FILL_COLOR = QColor(29, 78, 216, 38)
_AXIS_COLOR = QColor("#8792a8")
_GRID_COLOR = QColor("#e5e7eb")


class _ContinuousPlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self._mode: str = "pdf"
        self._title: str = ""
        self._grid: list[tuple[float, float]] = []
        self._marker_x: Optional[float] = None
        self._band: Optional[tuple[float, float]] = None

    def set_data(self, mode: str, title: str, grid: list[tuple[float, float]]) -> None:
        self._mode = mode
        self._title = title
        self._grid = list(grid)
        self.update()

    def set_marker(self, x: Optional[float]) -> None:
        self._marker_x = x
        self.update()

    def set_band(self, band: Optional[tuple[float, float]]) -> None:
        self._band = band
        self.update()

    def clear(self) -> None:
        self._grid = []
        self._marker_x = None
        self._band = None
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)

        w = self.width()
        h = self.height()
        margin_l, margin_r, margin_t, margin_b = 52, 20, 28, 32
        plot_w = max(40, w - margin_l - margin_r)
        plot_h = max(40, h - margin_t - margin_b)

        if not self._grid:
            painter.setPen(QPen(_AXIS_COLOR))
            painter.drawText(self.rect(), Qt.AlignCenter, "No posterior to display.")
            return

        xs = [x for x, _ in self._grid]
        ys = [y for _, y in self._grid]
        xmin, xmax = min(xs), max(xs)
        ymin = min(0.0, min(ys))
        ymax = max(ys) * 1.05 if max(ys) > 0 else 1.0
        if xmax == xmin:
            xmax = xmin + 1.0
        if ymax == ymin:
            ymax = ymin + 1.0

        def sx(x: float) -> float:
            return margin_l + (x - xmin) / (xmax - xmin) * plot_w

        def sy(y: float) -> float:
            return margin_t + plot_h - (y - ymin) / (ymax - ymin) * plot_h

        # Title
        painter.setPen(QPen(QColor("#1b2333")))
        tf = QFont()
        tf.setBold(True)
        tf.setPointSize(11)
        painter.setFont(tf)
        painter.drawText(
            QRectF(0, 4, w, 20), Qt.AlignCenter, self._title
        )

        # Grid (horizontal)
        painter.setPen(QPen(_GRID_COLOR))
        for i in range(1, 5):
            y = margin_t + plot_h * i / 5.0
            painter.drawLine(margin_l, int(y), margin_l + plot_w, int(y))

        # Axes
        painter.setPen(QPen(_AXIS_COLOR))
        painter.drawLine(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h)
        painter.drawLine(margin_l, margin_t, margin_l, margin_t + plot_h)

        # Axis labels (min/max x, 0 y and max y).
        lf = QFont()
        lf.setPointSize(9)
        painter.setFont(lf)
        painter.setPen(QPen(QColor("#4a5363")))
        painter.drawText(
            QRectF(margin_l - 4, margin_t + plot_h + 4, 60, 20),
            Qt.AlignLeft, f"{xmin:.4g}"
        )
        painter.drawText(
            QRectF(margin_l + plot_w - 60, margin_t + plot_h + 4, 60, 20),
            Qt.AlignRight, f"{xmax:.4g}"
        )
        painter.drawText(
            QRectF(4, margin_t - 6, margin_l - 8, 18),
            Qt.AlignRight, f"{ymax:.4g}"
        )
        painter.drawText(
            QRectF(4, margin_t + plot_h - 12, margin_l - 8, 18),
            Qt.AlignRight, f"{ymin:.4g}"
        )

        # Band shading (under the curve between a and b).
        if self._band is not None:
            a, b = self._band
            a_x = sx(max(xmin, min(xmax, a)))
            b_x = sx(max(xmin, min(xmax, b)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(_FILL_COLOR))
            painter.drawRect(QRectF(min(a_x, b_x), margin_t, abs(b_x - a_x), plot_h))

        # Curve
        pen = QPen(_PLOT_COLOR, 1.8)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        last = None
        for x, y in self._grid:
            pt_x = sx(x)
            pt_y = sy(y)
            if last is not None:
                painter.drawLine(last[0], last[1], pt_x, pt_y)
            last = (pt_x, pt_y)

        # Marker
        if self._marker_x is not None and xmin <= self._marker_x <= xmax:
            mx = sx(self._marker_x)
            pen = QPen(QColor("#b91c1c"), 1.2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(mx), margin_t, int(mx), margin_t + plot_h)


class ContinuousPosteriorPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._result: HybridResultDTO | None = None
        self._current: ContinuousPosteriorDTO | None = None
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Continuous posterior</b>"))
        header.addStretch()
        self._run_btn = QPushButton("Run Hybrid Query")
        self._run_btn.clicked.connect(self._viewmodel.run_hybrid)
        header.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._viewmodel.cancel_hybrid)
        header.addWidget(self._cancel_btn)
        layout.addLayout(header)

        self._diagnostics = QLabel("No hybrid result yet.")
        self._diagnostics.setStyleSheet(
            "color: #1b2333; background: #eef2ff; padding: 6px; border-radius: 4px;"
        )
        self._diagnostics.setWordWrap(True)
        layout.addWidget(self._diagnostics)

        self._error_banner = QFrame()
        self._error_banner.setStyleSheet(
            "background-color: #fde2e2; color: #8a1c1c; padding: 6px; border-radius: 4px;"
        )
        err_layout = QVBoxLayout(self._error_banner)
        err_layout.setContentsMargins(6, 4, 6, 4)
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        err_layout.addWidget(self._error_label)
        self._error_banner.setVisible(False)
        layout.addWidget(self._error_banner)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Node:"))
        self._node_combo = QComboBox()
        self._node_combo.currentTextChanged.connect(self._refresh_display)
        picker_row.addWidget(self._node_combo, stretch=1)
        self._pdf_radio = QRadioButton("PDF")
        self._cdf_radio = QRadioButton("CDF")
        self._pdf_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self._pdf_radio)
        group.addButton(self._cdf_radio)
        self._pdf_radio.toggled.connect(self._refresh_display)
        picker_row.addWidget(self._pdf_radio)
        picker_row.addWidget(self._cdf_radio)
        layout.addLayout(picker_row)

        stats_row = QHBoxLayout()
        self._stats_label = QLabel("—")
        self._stats_label.setStyleSheet("color: #4a5363; font-style: italic;")
        stats_row.addWidget(self._stats_label)
        stats_row.addStretch()
        layout.addLayout(stats_row)

        self._plot = _ContinuousPlotWidget()
        layout.addWidget(self._plot, stretch=1)

        query_form = QFormLayout()
        self._tail_spin = QDoubleSpinBox()
        self._tail_spin.setRange(-1e18, 1e18)
        self._tail_spin.setDecimals(6)
        self._tail_spin.valueChanged.connect(self._refresh_tail)
        query_form.addRow("Query x:", self._tail_spin)
        self._tail_label = QLabel("—")
        query_form.addRow("CDF(x) / tail:", self._tail_label)

        band_row = QHBoxLayout()
        self._band_a = QDoubleSpinBox()
        self._band_a.setRange(-1e18, 1e18)
        self._band_a.setDecimals(6)
        self._band_a.valueChanged.connect(self._refresh_tail)
        self._band_b = QDoubleSpinBox()
        self._band_b.setRange(-1e18, 1e18)
        self._band_b.setDecimals(6)
        self._band_b.setValue(1.0)
        self._band_b.valueChanged.connect(self._refresh_tail)
        band_row.addWidget(QLabel("a"))
        band_row.addWidget(self._band_a)
        band_row.addWidget(QLabel("b"))
        band_row.addWidget(self._band_b)
        query_form.addRow("Band [a, b]:", self._pack(band_row))
        self._band_label = QLabel("—")
        query_form.addRow("P(a ≤ X ≤ b):", self._band_label)

        self._quantile_spin = QDoubleSpinBox()
        self._quantile_spin.setRange(0.001, 0.999)
        self._quantile_spin.setDecimals(3)
        self._quantile_spin.setSingleStep(0.05)
        self._quantile_spin.setValue(0.95)
        self._quantile_spin.valueChanged.connect(self._refresh_quantile)
        query_form.addRow("Quantile q:", self._quantile_spin)
        self._quantile_label = QLabel("—")
        query_form.addRow("x such that CDF(x)=q:", self._quantile_label)

        layout.addLayout(query_form)

    @staticmethod
    def _pack(sublayout: QHBoxLayout) -> QWidget:
        w = QWidget()
        w.setLayout(sublayout)
        return w

    def _bind_viewmodel(self) -> None:
        self._viewmodel.hybrid_started.connect(self._on_started)
        self._viewmodel.hybrid_finished.connect(self._on_finished)
        self._viewmodel.hybrid_failed.connect(self._on_failed)
        self._viewmodel.busy_changed.connect(self._on_busy_changed)
        self._viewmodel.continuous_nodes_changed.connect(
            lambda *_: self._run_btn.setEnabled(
                not self._viewmodel.is_busy and bool(self._viewmodel.continuous_nodes)
            )
        )

    def _on_busy_changed(self, busy: bool) -> None:
        self._run_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)

    # ----------------------------------------------------------- events

    def _on_started(self) -> None:
        self._diagnostics.setText("Running hybrid inference…")
        self._error_banner.setVisible(False)
        self._plot.clear()
        self._node_combo.clear()

    def _on_finished(self, result: HybridResultDTO) -> None:
        self._result = result
        self._diagnostics.setText(
            f"Iterations: {result.iterations_used} / {result.max_iters}   ·   "
            f"max error: {result.final_max_error:.4g}   ·   "
            f"converged: {'yes' if result.converged else 'no'}   ·   "
            f"continuous: {len(result.continuous)}   ·   "
            f"discrete: {len(result.discrete)}"
        )
        self._error_banner.setVisible(False)
        self._node_combo.blockSignals(True)
        self._node_combo.clear()
        for name in result.continuous.keys():
            self._node_combo.addItem(name)
        self._node_combo.blockSignals(False)
        self._refresh_display()

    def _on_failed(self, message: str) -> None:
        self._result = None
        self._current = None
        self._error_label.setText(message)
        self._error_banner.setVisible(True)
        self._diagnostics.setText("Hybrid query failed.")
        self._plot.clear()
        self._node_combo.clear()

    # ----------------------------------------------------------- rendering

    def _refresh_display(self) -> None:
        if self._result is None:
            return
        name = self._node_combo.currentText()
        if not name:
            self._current = None
            self._plot.clear()
            self._stats_label.setText("—")
            return
        dto = self._result.continuous.get(name)
        if dto is None:
            return
        self._current = dto
        mode = "pdf" if self._pdf_radio.isChecked() else "cdf"
        grid = list(dto.pdf_grid) if mode == "pdf" else list(dto.cdf_grid)
        self._plot.set_data(mode, f"{mode.upper()} — {name}", grid)
        self._stats_label.setText(
            f"support=[{dto.support[0]:.4g}, {dto.support[1]:.4g}]   "
            f"bins={dto.num_bins}   "
            f"mean={dto.mean:.4g}   std={dto.std:.4g}   median={dto.median:.4g}"
        )
        lo, hi = dto.support
        self._tail_spin.setMinimum(lo - abs(lo) - 1.0)
        self._tail_spin.setMaximum(hi + abs(hi) + 1.0)
        self._band_a.setMinimum(lo - abs(lo) - 1.0)
        self._band_a.setMaximum(hi + abs(hi) + 1.0)
        self._band_b.setMinimum(lo - abs(lo) - 1.0)
        self._band_b.setMaximum(hi + abs(hi) + 1.0)
        if not (lo <= self._tail_spin.value() <= hi):
            self._tail_spin.setValue((lo + hi) / 2.0)
        if not (lo <= self._band_a.value() <= hi):
            self._band_a.setValue(lo)
        if not (lo <= self._band_b.value() <= hi):
            self._band_b.setValue(hi)
        self._refresh_tail()
        self._refresh_quantile()

    def _refresh_tail(self) -> None:
        if self._current is None:
            return
        x = float(self._tail_spin.value())
        cdf = self._interp(self._current.cdf_grid, x)
        self._tail_label.setText(
            f"CDF(x) = {cdf:.6f}   ·   P(X > x) = {1 - cdf:.6f}   ·   P(X < x) = {cdf:.6f}"
        )
        a = float(self._band_a.value())
        b = float(self._band_b.value())
        if b < a:
            a, b = b, a
        p = max(0.0, self._interp(self._current.cdf_grid, b) - self._interp(self._current.cdf_grid, a))
        self._band_label.setText(f"{p:.6f}")
        self._plot.set_marker(x)
        self._plot.set_band((a, b))

    def _refresh_quantile(self) -> None:
        if self._current is None:
            return
        q = float(self._quantile_spin.value())
        # Pre-sampled quantiles first, then fall back to CDF inversion.
        match = next((qv for qp, qv in self._current.quantiles if abs(qp - q) < 1e-6), None)
        if match is None:
            match = self._invert_cdf(self._current.cdf_grid, q)
        self._quantile_label.setText(f"{match:.6f}")

    @staticmethod
    def _interp(grid: tuple[tuple[float, float], ...], x: float) -> float:
        if not grid:
            return 0.0
        if x <= grid[0][0]:
            return grid[0][1]
        if x >= grid[-1][0]:
            return grid[-1][1]
        for (x0, y0), (x1, y1) in zip(grid, grid[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return grid[-1][1]

    @staticmethod
    def _invert_cdf(grid: tuple[tuple[float, float], ...], q: float) -> float:
        q = max(0.0, min(1.0, q))
        for (x0, y0), (x1, y1) in zip(grid, grid[1:]):
            if y0 <= q <= y1:
                if y1 == y0:
                    return x0
                t = (q - y0) / (y1 - y0)
                return x0 + t * (x1 - x0)
        return grid[-1][0]
