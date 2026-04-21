"""Results panel — posterior bar chart + error banner."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from pybncore_gui.domain.results import PosteriorResult
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.panels.bar_chart import BarChartWidget


class ResultsPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._header = QLabel("Posterior")
        self._header.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._header)

        self._evidence_label = QLabel("Evidence: (none)")
        self._evidence_label.setStyleSheet("color: #4a5363;")
        self._evidence_label.setWordWrap(True)
        layout.addWidget(self._evidence_label)

        self._error_banner = QFrame(self)
        self._error_banner.setFrameShape(QFrame.StyledPanel)
        self._error_banner.setStyleSheet(
            "background-color: #fde2e2; color: #8a1c1c; padding: 6px; border-radius: 4px;"
        )
        self._error_text = QLabel("")
        self._error_text.setWordWrap(True)
        err_layout = QVBoxLayout(self._error_banner)
        err_layout.setContentsMargins(6, 4, 6, 4)
        err_layout.addWidget(self._error_text)
        self._error_banner.setVisible(False)
        layout.addWidget(self._error_banner)

        self._chart = BarChartWidget(self)
        layout.addWidget(self._chart, stretch=1)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.query_started.connect(self._on_query_started)
        self._viewmodel.query_finished.connect(self._on_query_finished)
        self._viewmodel.query_failed.connect(self._on_query_failed)
        self._viewmodel.model_cleared.connect(self._reset)
        self._viewmodel.model_loaded.connect(lambda *_: self._reset())

    def _on_query_started(self, node_id: str) -> None:
        self._header.setText(f"Running query on '{node_id}'…")
        self._error_banner.setVisible(False)
        self._chart.clear()

    def _on_query_finished(self, result: PosteriorResult) -> None:
        self._header.setText(f"Posterior — {result.node}")
        evidence = result.evidence_snapshot
        if evidence:
            txt = "Evidence: " + ", ".join(f"{k}={v}" for k, v in evidence.items())
        else:
            txt = "Evidence: (none)"
        self._evidence_label.setText(txt)
        self._chart.set_data(
            title=f"P({result.node} | evidence)",
            pairs=list(zip(result.states, result.probabilities)),
        )
        self._error_banner.setVisible(False)

    def _on_query_failed(self, message: str) -> None:
        self._header.setText("Posterior — failed")
        self._error_text.setText(message)
        self._error_banner.setVisible(True)
        self._chart.clear()

    def _reset(self) -> None:
        self._header.setText("Posterior")
        self._evidence_label.setText("Evidence: (none)")
        self._error_banner.setVisible(False)
        self._chart.clear()
