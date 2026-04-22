"""Monte-Carlo workflow — sample evidence, run batch, aggregate posteriors."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import MonteCarloResult
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.panels.bar_chart import BarChartWidget


class MonteCarloPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._last: MonteCarloResult | None = None
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel("<b>Monte Carlo</b>"))
        layout.addWidget(
            QLabel(
                "Samples evidence rows uniformly from each observed node's states, "
                "runs a batch query, and reports mean/std of the posterior.",
                wordWrap=True,
            )
        )

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Query nodes:"))
        self._query_list = QListWidget()
        self._query_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._query_list.setFixedHeight(90)
        row1.addWidget(self._query_list, stretch=1)
        row1.addWidget(QLabel("Observed:"))
        self._observed_list = QListWidget()
        self._observed_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._observed_list.setFixedHeight(90)
        row1.addWidget(self._observed_list, stretch=1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Samples:"))
        self._samples_spin = QSpinBox()
        self._samples_spin.setRange(1, 1_000_000)
        self._samples_spin.setValue(500)
        row2.addWidget(self._samples_spin)
        row2.addWidget(QLabel("Seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2**31 - 1)
        self._seed_spin.setValue(42)
        row2.addWidget(self._seed_spin)
        self._run_btn = QPushButton("Run Monte Carlo")
        self._run_btn.clicked.connect(self._run)
        row2.addWidget(self._run_btn)
        layout.addLayout(row2)

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

        results_row = QHBoxLayout()
        results_row.addWidget(QLabel("Display:"))
        self._node_combo = QComboBox()
        self._node_combo.currentTextChanged.connect(self._refresh_display)
        results_row.addWidget(self._node_combo)
        results_row.addStretch()
        layout.addLayout(results_row)

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["State", "Mean", "Std"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._chart = BarChartWidget()
        layout.addWidget(self._chart, stretch=1)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.model_loaded.connect(self._refresh_nodes)
        self._viewmodel.structure_changed.connect(self._refresh_nodes)
        self._viewmodel.monte_carlo_started.connect(self._on_started)
        self._viewmodel.monte_carlo_finished.connect(self._on_finished)
        self._viewmodel.monte_carlo_failed.connect(self._on_failed)
        self._viewmodel.busy_changed.connect(lambda b: self._run_btn.setEnabled(not b))

    def _refresh_nodes(self, *_args) -> None:
        nodes = [n.id for n in self._viewmodel.model_service.list_nodes() if n.states]
        self._query_list.clear()
        self._observed_list.clear()
        for n in sorted(nodes):
            self._query_list.addItem(n)
            self._observed_list.addItem(n)

    def _run(self) -> None:
        query = [i.text() for i in self._query_list.selectedItems()]
        observed = [i.text() for i in self._observed_list.selectedItems()]
        if not query:
            self._error_label.setText("Select at least one query node.")
            self._error_banner.setVisible(True)
            return
        self._error_banner.setVisible(False)
        self._viewmodel.run_monte_carlo(
            query, observed, int(self._samples_spin.value()), int(self._seed_spin.value())
        )

    def _on_started(self) -> None:
        self._error_banner.setVisible(False)
        self._table.setRowCount(0)
        self._chart.clear()
        self._node_combo.clear()

    def _on_finished(self, result: MonteCarloResult) -> None:
        self._last = result
        self._node_combo.blockSignals(True)
        self._node_combo.clear()
        for node in result.query_nodes:
            self._node_combo.addItem(node)
        self._node_combo.blockSignals(False)
        self._refresh_display()

    def _on_failed(self, message: str) -> None:
        self._last = None
        self._error_label.setText(message)
        self._error_banner.setVisible(True)
        self._table.setRowCount(0)
        self._chart.clear()
        self._node_combo.clear()

    def _refresh_display(self) -> None:
        if self._last is None:
            return
        node = self._node_combo.currentText()
        if not node or node not in self._last.summaries:
            return
        summary = self._last.summaries[node]
        self._table.setRowCount(0)
        for state, mean, std in zip(summary.states, summary.mean, summary.std):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(state))
            mean_item = QTableWidgetItem(f"{mean:.4f}")
            mean_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 1, mean_item)
            std_item = QTableWidgetItem(f"{std:.4f}")
            std_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 2, std_item)
        self._chart.set_data(
            f"Mean posterior for '{node}' over {self._last.num_samples} samples",
            list(zip(summary.states, summary.mean)),
        )
