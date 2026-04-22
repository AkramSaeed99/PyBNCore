"""Value of Information — ranked candidate observations."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import VOIReport
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.panels.bar_chart import BarChartWidget


class VOIPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel("<b>Value of information</b>"))

        form = QFormLayout()
        self._query_combo = QComboBox()
        form.addRow("Query node:", self._query_combo)
        layout.addLayout(form)

        self._all_candidates = QCheckBox("Use all other nodes as candidates")
        self._all_candidates.setChecked(True)
        self._all_candidates.toggled.connect(self._toggle_candidates_list)
        layout.addWidget(self._all_candidates)

        self._candidate_list = QListWidget()
        self._candidate_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._candidate_list.setFixedHeight(120)
        self._candidate_list.setEnabled(False)
        layout.addWidget(self._candidate_list)

        actions = QHBoxLayout()
        self._run_btn = QPushButton("Run VOI")
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

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Candidate", "VOI"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._chart = BarChartWidget()
        layout.addWidget(self._chart, stretch=1)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.model_loaded.connect(self._refresh_nodes)
        self._viewmodel.structure_changed.connect(self._refresh_nodes)
        self._viewmodel.voi_started.connect(self._on_started)
        self._viewmodel.voi_finished.connect(self._on_finished)
        self._viewmodel.voi_failed.connect(self._on_failed)
        self._viewmodel.busy_changed.connect(lambda b: self._run_btn.setEnabled(not b))

    def _refresh_nodes(self, *_args) -> None:
        nodes = self._viewmodel.model_service.list_nodes()
        current_query = self._query_combo.currentText()
        selected_candidates = {
            i.text() for i in self._candidate_list.selectedItems()
        }
        self._query_combo.blockSignals(True)
        self._query_combo.clear()
        for node in sorted(nodes, key=lambda n: n.id):
            if node.states:
                self._query_combo.addItem(node.id)
        idx = self._query_combo.findText(current_query)
        if idx >= 0:
            self._query_combo.setCurrentIndex(idx)
        self._query_combo.blockSignals(False)

        self._candidate_list.clear()
        for node in sorted(nodes, key=lambda n: n.id):
            if node.states:
                self._candidate_list.addItem(node.id)
        for i in range(self._candidate_list.count()):
            if self._candidate_list.item(i).text() in selected_candidates:
                self._candidate_list.item(i).setSelected(True)

    def _toggle_candidates_list(self, checked: bool) -> None:
        self._candidate_list.setEnabled(not checked)

    def _run(self) -> None:
        query = self._query_combo.currentText()
        if not query:
            return
        candidates = None
        if not self._all_candidates.isChecked():
            candidates = [i.text() for i in self._candidate_list.selectedItems()]
            if not candidates:
                self._error_label.setText("Select at least one candidate.")
                self._error_banner.setVisible(True)
                return
        self._viewmodel.run_voi(query, candidates)

    def _on_started(self) -> None:
        self._error_banner.setVisible(False)
        self._table.setRowCount(0)
        self._chart.clear()

    def _on_finished(self, report: VOIReport) -> None:
        self._table.setRowCount(0)
        for entry in report.entries:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(entry.candidate))
            score_item = QTableWidgetItem(f"{entry.score:.6f}")
            score_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 1, score_item)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        top = list(report.entries)[:15]
        self._chart.set_data(
            f"VOI for '{report.query_node}'",
            [(e.candidate, float(e.score)) for e in top],
        )

    def _on_failed(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_banner.setVisible(True)
        self._table.setRowCount(0)
        self._chart.clear()
