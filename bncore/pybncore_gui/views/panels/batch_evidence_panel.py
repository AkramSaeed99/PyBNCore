"""Batch evidence editor and results matrix."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.node import NodeModel
from pybncore_gui.domain.results import BatchQueryResult
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_NONE_LABEL = "(none)"


class BatchEvidencePanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._nodes: list[NodeModel] = []
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(QLabel("<b>Batch query</b>"))

        splitter = QSplitter(Qt.Vertical, self)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Query nodes:"))
        self._query_list = QListWidget()
        self._query_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._query_list.setFixedHeight(90)
        config_row.addWidget(self._query_list, stretch=1)

        config_col = QVBoxLayout()
        config_col.addWidget(QLabel("Rows:"))
        self._row_spin = QSpinBox()
        self._row_spin.setRange(1, 1000)
        self._row_spin.setValue(3)
        config_col.addWidget(self._row_spin)
        self._refresh_btn = QPushButton("Rebuild evidence table")
        self._refresh_btn.clicked.connect(self._rebuild_evidence_table)
        config_col.addWidget(self._refresh_btn)
        config_col.addStretch()
        config_row.addLayout(config_col)
        top_layout.addLayout(config_row)

        top_layout.addWidget(QLabel("Evidence rows (columns are all nodes; leave '(none)' for unobserved):"))
        self._evidence_table = QTableWidget(0, 0, self)
        self._evidence_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        top_layout.addWidget(self._evidence_table, stretch=1)

        button_row = QHBoxLayout()
        self._run_btn = QPushButton("Run batch query")
        self._run_btn.clicked.connect(self._run)
        button_row.addStretch()
        button_row.addWidget(self._run_btn)
        top_layout.addLayout(button_row)

        splitter.addWidget(top)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(QLabel("<b>Results</b>"))

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
        bottom_layout.addWidget(self._error_banner)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Show posteriors for:"))
        self._node_combo = QComboBox()
        self._node_combo.currentTextChanged.connect(self._refresh_result_view)
        picker_row.addWidget(self._node_combo)
        picker_row.addStretch()
        bottom_layout.addLayout(picker_row)

        self._result_table = QTableWidget(0, 0, self)
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._result_table.setAlternatingRowColors(True)
        bottom_layout.addWidget(self._result_table, stretch=1)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter, stretch=1)

        self._last_result: BatchQueryResult | None = None

    def _bind_viewmodel(self) -> None:
        self._viewmodel.model_loaded.connect(self._on_model_loaded)
        self._viewmodel.structure_changed.connect(self._on_model_loaded)
        self._viewmodel.batch_started.connect(self._on_started)
        self._viewmodel.batch_finished.connect(self._on_finished)
        self._viewmodel.batch_failed.connect(self._on_failed)
        self._viewmodel.busy_changed.connect(lambda b: self._run_btn.setEnabled(not b))

    # ------------------------------------------------------------- build

    def _on_model_loaded(self, nodes=None, edges=None) -> None:
        self._nodes = self._viewmodel.model_service.list_nodes()
        self._query_list.clear()
        for n in sorted(self._nodes, key=lambda x: x.id):
            self._query_list.addItem(n.id)
        self._rebuild_evidence_table()

    def _rebuild_evidence_table(self) -> None:
        nodes = [n for n in self._nodes if n.states]
        self._evidence_table.clear()
        self._evidence_table.setColumnCount(len(nodes))
        self._evidence_table.setRowCount(self._row_spin.value())
        self._evidence_table.setHorizontalHeaderLabels([n.id for n in nodes])
        for col, node in enumerate(nodes):
            for row in range(self._evidence_table.rowCount()):
                combo = QComboBox()
                combo.addItem(_NONE_LABEL, userData=None)
                for state in node.states:
                    combo.addItem(state, userData=state)
                self._evidence_table.setCellWidget(row, col, combo)
        self._evidence_table.resizeColumnsToContents()

    # ------------------------------------------------------------- run

    def _collect_rows(self) -> list[dict[str, str]]:
        nodes = [n for n in self._nodes if n.states]
        rows: list[dict[str, str]] = []
        for row in range(self._evidence_table.rowCount()):
            row_evidence: dict[str, str] = {}
            for col, node in enumerate(nodes):
                combo = self._evidence_table.cellWidget(row, col)
                if isinstance(combo, QComboBox):
                    val = combo.currentData()
                    if val is not None:
                        row_evidence[node.id] = val
            rows.append(row_evidence)
        return rows

    def _selected_query_nodes(self) -> list[str]:
        return [item.text() for item in self._query_list.selectedItems()]

    def _run(self) -> None:
        query_nodes = self._selected_query_nodes()
        if not query_nodes:
            QMessageBox.warning(self, "Batch query", "Select at least one query node.")
            return
        rows = self._collect_rows()
        if not rows:
            QMessageBox.warning(self, "Batch query", "Evidence table is empty.")
            return
        self._error_banner.setVisible(False)
        self._viewmodel.run_batch(query_nodes, rows)

    # ------------------------------------------------------------- results

    def _on_started(self) -> None:
        self._error_banner.setVisible(False)

    def _on_finished(self, result: BatchQueryResult) -> None:
        self._last_result = result
        self._node_combo.blockSignals(True)
        self._node_combo.clear()
        for node in result.nodes:
            self._node_combo.addItem(node)
        self._node_combo.blockSignals(False)
        self._refresh_result_view()

    def _on_failed(self, message: str) -> None:
        self._last_result = None
        self._error_label.setText(message)
        self._error_banner.setVisible(True)
        self._result_table.setRowCount(0)
        self._result_table.setColumnCount(0)

    def _refresh_result_view(self) -> None:
        result = self._last_result
        if result is None:
            return
        node = self._node_combo.currentText()
        if not node or node not in result.marginals:
            return
        matrix = np.asarray(result.marginals[node], dtype=np.float64)
        states = list(result.state_labels.get(node, tuple()))
        self._result_table.clear()
        self._result_table.setRowCount(matrix.shape[0])
        self._result_table.setColumnCount(matrix.shape[1])
        self._result_table.setHorizontalHeaderLabels(states)
        vlabels = [f"row {i}" for i in range(matrix.shape[0])]
        self._result_table.setVerticalHeaderLabels(vlabels)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                item = QTableWidgetItem(f"{matrix[r, c]:.4f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                # Heatmap: lightness proportional to probability.
                p = float(matrix[r, c])
                shade = int(max(0, min(1, p)) * 150)
                item.setBackground(QColor(60 + shade, 120 + shade, 200 + min(55, shade)))
                item.setForeground(QColor("#1b2333"))
                self._result_table.setItem(r, c, item)
        self._result_table.resizeColumnsToContents()
