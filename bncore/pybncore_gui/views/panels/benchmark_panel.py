"""Performance benchmark — time batch queries over a sweep of row counts."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import BenchmarkResult
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.panels.bar_chart import BarChartWidget


class BenchmarkPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel("<b>Performance benchmark</b>"))

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
        row2.addWidget(QLabel("Row counts (csv):"))
        self._rows_edit = QLineEdit("10, 50, 100, 500, 1000")
        row2.addWidget(self._rows_edit, stretch=1)
        row2.addWidget(QLabel("Seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2**31 - 1)
        self._seed_spin.setValue(42)
        row2.addWidget(self._seed_spin)
        self._run_btn = QPushButton("Run benchmark")
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

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Rows", "Elapsed (ms)", "ms / row"])
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
        self._viewmodel.benchmark_started.connect(self._on_started)
        self._viewmodel.benchmark_finished.connect(self._on_finished)
        self._viewmodel.benchmark_failed.connect(self._on_failed)
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
        raw = [x.strip() for x in self._rows_edit.text().split(",") if x.strip()]
        try:
            row_counts = [int(x) for x in raw]
        except ValueError:
            self._error_label.setText("Row counts must be integers separated by commas.")
            self._error_banner.setVisible(True)
            return
        if not query:
            self._error_label.setText("Select at least one query node.")
            self._error_banner.setVisible(True)
            return
        if not row_counts:
            self._error_label.setText("Provide at least one row count.")
            self._error_banner.setVisible(True)
            return
        self._error_banner.setVisible(False)
        self._viewmodel.run_benchmark(query, observed, row_counts, int(self._seed_spin.value()))

    def _on_started(self) -> None:
        self._error_banner.setVisible(False)
        self._table.setRowCount(0)
        self._chart.clear()

    def _on_finished(self, result: BenchmarkResult) -> None:
        self._table.setRowCount(0)
        for pt in result.points:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(pt.num_rows)))
            ms_item = QTableWidgetItem(f"{pt.elapsed_ms:.2f}")
            ms_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 1, ms_item)
            per_item = QTableWidgetItem(f"{pt.ms_per_row:.4f}")
            per_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 2, per_item)
        self._chart.set_data(
            "Elapsed (ms) per batch size",
            [(str(pt.num_rows), pt.elapsed_ms) for pt in result.points],
        )

    def _on_failed(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_banner.setVisible(True)
        self._table.setRowCount(0)
        self._chart.clear()
