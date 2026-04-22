"""Sensitivity analysis — ranked parameter influences on a query."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import SensitivityReport
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.panels.bar_chart import BarChartWidget


class SensitivityPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel("<b>Parameter sensitivity</b>"))

        form = QFormLayout()
        self._query_combo = QComboBox()
        self._query_combo.currentTextChanged.connect(self._refresh_states)
        self._state_combo = QComboBox()
        self._n_top_spin = QSpinBox()
        self._n_top_spin.setRange(1, 500)
        self._n_top_spin.setValue(10)
        self._epsilon_spin = QDoubleSpinBox()
        self._epsilon_spin.setRange(0.0005, 0.5)
        self._epsilon_spin.setSingleStep(0.005)
        self._epsilon_spin.setDecimals(4)
        self._epsilon_spin.setValue(0.05)
        form.addRow("Query node:", self._query_combo)
        form.addRow("Query state:", self._state_combo)
        form.addRow("Top N:", self._n_top_spin)
        form.addRow("Perturbation ε:", self._epsilon_spin)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run)
        actions.addStretch()
        actions.addWidget(self._run_btn)
        layout.addLayout(actions)

        self._error_banner = QFrame()
        self._error_banner.setStyleSheet(
            "background-color: #fde2e2; color: #8a1c1c; padding: 6px; border-radius: 4px;"
        )
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        err_layout = QVBoxLayout(self._error_banner)
        err_layout.setContentsMargins(6, 4, 6, 4)
        err_layout.addWidget(self._error_label)
        self._error_banner.setVisible(False)
        layout.addWidget(self._error_banner)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(
            ["Target node", "Parent config", "Target state", "Score"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._chart = BarChartWidget()
        layout.addWidget(self._chart, stretch=1)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.model_loaded.connect(self._refresh_nodes)
        self._viewmodel.structure_changed.connect(self._refresh_nodes)
        self._viewmodel.sensitivity_started.connect(self._on_started)
        self._viewmodel.sensitivity_finished.connect(self._on_finished)
        self._viewmodel.sensitivity_failed.connect(self._on_failed)
        self._viewmodel.busy_changed.connect(lambda b: self._run_btn.setEnabled(not b))

    def _refresh_nodes(self, *_args) -> None:
        current = self._query_combo.currentText()
        nodes = self._viewmodel.model_service.list_nodes()
        self._query_combo.blockSignals(True)
        self._query_combo.clear()
        for node in sorted(nodes, key=lambda n: n.id):
            if node.states:
                self._query_combo.addItem(node.id)
        idx = self._query_combo.findText(current)
        if idx >= 0:
            self._query_combo.setCurrentIndex(idx)
        self._query_combo.blockSignals(False)
        self._refresh_states(self._query_combo.currentText())

    def _refresh_states(self, node_id: str) -> None:
        self._state_combo.clear()
        if not node_id:
            return
        try:
            states = self._viewmodel.model_service.get_outcomes(node_id)
        except Exception:
            states = []
        for s in states:
            self._state_combo.addItem(s)

    def _run(self) -> None:
        q_node = self._query_combo.currentText()
        q_state = self._state_combo.currentText()
        if not q_node or not q_state:
            return
        self._viewmodel.run_sensitivity(
            q_node,
            q_state,
            int(self._n_top_spin.value()),
            float(self._epsilon_spin.value()),
        )

    def _on_started(self) -> None:
        self._error_banner.setVisible(False)
        self._table.setRowCount(0)
        self._chart.clear()

    def _on_finished(self, report: SensitivityReport) -> None:
        self._table.setRowCount(0)
        for entry in report.entries:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(entry.target_node))
            self._table.setItem(row, 1, QTableWidgetItem(", ".join(entry.parent_config) or "—"))
            self._table.setItem(row, 2, QTableWidgetItem(entry.target_state))
            score_item = QTableWidgetItem(f"{entry.score:.6f}")
            score_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 3, score_item)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        top_entries = list(report.entries)[:10]
        pairs = [
            (f"{e.target_node}[{e.target_state}]", abs(e.score)) for e in top_entries
        ]
        self._chart.set_data(
            f"Top sensitivities for P({report.query_node}={report.query_state})",
            pairs,
        )

    def _on_failed(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_banner.setVisible(True)
        self._table.setRowCount(0)
        self._chart.clear()
