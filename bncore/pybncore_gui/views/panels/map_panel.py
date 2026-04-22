"""MAP / MPE result panel."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import MAPResult
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel


class MAPPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>MAP / MPE</b>"))
        header.addStretch()
        self._run_btn = QPushButton("Run MAP")
        self._run_btn.clicked.connect(self._viewmodel.run_map)
        header.addWidget(self._run_btn)
        layout.addLayout(header)

        self._summary = QLabel("Click 'Run MAP' to compute the most-probable assignment.")
        self._summary.setStyleSheet("color: #4a5363;")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._error = QFrame(self)
        self._error.setStyleSheet(
            "background-color: #fde2e2; color: #8a1c1c; padding: 6px; border-radius: 4px;"
        )
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        err_layout = QVBoxLayout(self._error)
        err_layout.setContentsMargins(6, 4, 6, 4)
        err_layout.addWidget(self._error_label)
        self._error.setVisible(False)
        layout.addWidget(self._error)

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Node", "MAP state"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.map_started.connect(self._on_started)
        self._viewmodel.map_finished.connect(self._on_finished)
        self._viewmodel.map_failed.connect(self._on_failed)
        self._viewmodel.model_loaded.connect(lambda *_: self._reset())
        self._viewmodel.structure_changed.connect(lambda *_: self._reset())
        self._viewmodel.busy_changed.connect(lambda b: self._run_btn.setEnabled(not b))

    def _on_started(self) -> None:
        self._summary.setText("Running MAP query…")
        self._error.setVisible(False)
        self._table.setRowCount(0)

    def _on_finished(self, result: MAPResult) -> None:
        self._summary.setText(
            f"MAP assignment over {len(result.assignment)} nodes. "
            f"Evidence: {len(result.evidence_snapshot)} hard"
            + (
                f", soft on {len(result.soft_evidence_snapshot)} node(s)."
                if result.soft_evidence_snapshot
                else "."
            )
        )
        self._table.setRowCount(0)
        for node, state in sorted(result.assignment.items()):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(node))
            value_item = QTableWidgetItem(state)
            if node in result.evidence_snapshot:
                value_item.setToolTip("Observed (evidence)")
            self._table.setItem(row, 1, value_item)
        self._error.setVisible(False)

    def _on_failed(self, message: str) -> None:
        self._summary.setText("MAP failed.")
        self._error_label.setText(message)
        self._error.setVisible(True)
        self._table.setRowCount(0)

    def _reset(self) -> None:
        self._table.setRowCount(0)
        self._summary.setText("Click 'Run MAP' to compute the most-probable assignment.")
        self._error.setVisible(False)
