"""Side-by-side comparison of two posteriors from the query history."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import PosteriorResult
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.panels.bar_chart import BarChartWidget


class ComparePanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._history: list[PosteriorResult] = []
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Compare posteriors</b>"))
        layout.addWidget(
            QLabel(
                "Each single-node query adds an entry to history. Pick two to compare.",
                wordWrap=True,
            )
        )

        selectors = QFormLayout()
        self._left_combo = QComboBox()
        self._right_combo = QComboBox()
        self._left_combo.currentIndexChanged.connect(self._refresh)
        self._right_combo.currentIndexChanged.connect(self._refresh)
        selectors.addRow("Left:", self._left_combo)
        selectors.addRow("Right:", self._right_combo)
        layout.addLayout(selectors)

        actions = QHBoxLayout()
        self._use_latest_btn = QPushButton("Use latest two")
        self._use_latest_btn.clicked.connect(self._use_latest_two)
        self._clear_btn = QPushButton("Clear history")
        self._clear_btn.clicked.connect(self._clear_history)
        actions.addWidget(self._use_latest_btn)
        actions.addWidget(self._clear_btn)
        actions.addStretch()
        layout.addLayout(actions)

        self._splitter = QSplitter(Qt.Horizontal, self)
        self._left_chart = BarChartWidget()
        self._right_chart = BarChartWidget()
        self._splitter.addWidget(self._chart_container("Left", self._left_chart))
        self._splitter.addWidget(self._chart_container("Right", self._right_chart))
        layout.addWidget(self._splitter, stretch=1)

        self._diff_label = QLabel("")
        self._diff_label.setStyleSheet("color: #4a5363; font-style: italic;")
        self._diff_label.setWordWrap(True)
        layout.addWidget(self._diff_label)

    @staticmethod
    def _chart_container(title: str, chart: BarChartWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"<b>{title}</b>")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        layout.addWidget(chart, stretch=1)
        return container

    def _bind_viewmodel(self) -> None:
        self._viewmodel.posterior_history_changed.connect(self._populate)
        self._viewmodel.model_loaded.connect(lambda *_: self._populate([]))

    def _populate(self, history: list[PosteriorResult]) -> None:
        self._history = list(history)
        self._left_combo.blockSignals(True)
        self._right_combo.blockSignals(True)
        self._left_combo.clear()
        self._right_combo.clear()
        for i, r in enumerate(self._history):
            label = self._label(i, r)
            self._left_combo.addItem(label, userData=i)
            self._right_combo.addItem(label, userData=i)
        if len(self._history) >= 2:
            self._left_combo.setCurrentIndex(len(self._history) - 2)
            self._right_combo.setCurrentIndex(len(self._history) - 1)
        self._left_combo.blockSignals(False)
        self._right_combo.blockSignals(False)
        self._refresh()

    @staticmethod
    def _label(index: int, r: PosteriorResult) -> str:
        ev = ",".join(f"{k}={v}" for k, v in list(r.evidence_snapshot.items())[:3])
        if len(r.evidence_snapshot) > 3:
            ev += "…"
        return f"#{index + 1} {r.node} ({r.engine_label})" + (f" | {ev}" if ev else "")

    def _refresh(self) -> None:
        left = self._current(self._left_combo)
        right = self._current(self._right_combo)
        self._left_chart.clear()
        self._right_chart.clear()
        if left is not None:
            self._left_chart.set_data(
                f"P({left.node} | evidence)",
                list(zip(left.states, left.probabilities)),
            )
        if right is not None:
            self._right_chart.set_data(
                f"P({right.node} | evidence)",
                list(zip(right.states, right.probabilities)),
            )
        if left is not None and right is not None and left.node == right.node:
            diffs = [
                (s, rp - lp)
                for s, lp, rp in zip(left.states, left.probabilities, right.probabilities)
            ]
            parts = [f"{s}: {d:+.4f}" for s, d in diffs]
            self._diff_label.setText("Δ (Right − Left)   " + "   ".join(parts))
        elif left is not None and right is not None:
            self._diff_label.setText(
                f"Comparing different nodes ('{left.node}' vs '{right.node}')."
            )
        else:
            self._diff_label.setText("")

    def _current(self, combo: QComboBox) -> PosteriorResult | None:
        idx = combo.currentData()
        if idx is None or not isinstance(idx, int):
            return None
        if 0 <= idx < len(self._history):
            return self._history[idx]
        return None

    def _use_latest_two(self) -> None:
        if len(self._history) >= 2:
            self._left_combo.setCurrentIndex(len(self._history) - 2)
            self._right_combo.setCurrentIndex(len(self._history) - 1)

    def _clear_history(self) -> None:
        # History lives on the viewmodel; we can't mutate it directly. Clear visually.
        self._populate([])
