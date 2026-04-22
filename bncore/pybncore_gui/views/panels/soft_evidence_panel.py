"""Soft-evidence editor — likelihood vector per node."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.viewmodels.main_viewmodel import MainViewModel


class SoftEvidencePanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._current_node: Optional[str] = None
        self._build_ui()
        self._bind_viewmodel()
        self._reset()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Soft / virtual evidence</b>"))
        header.addStretch()
        self._clear_all_btn = QPushButton("Clear all")
        self._clear_all_btn.clicked.connect(self._viewmodel.clear_soft_evidence)
        header.addWidget(self._clear_all_btn)
        layout.addLayout(header)

        self._node_label = QLabel("Select a node to edit its likelihood vector.")
        self._node_label.setWordWrap(True)
        self._node_label.setStyleSheet("color: #4a5363;")
        layout.addWidget(self._node_label)

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["State", "Likelihood"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        layout.addWidget(self._table, stretch=1)

        actions = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        self._uniform_btn = QPushButton("Reset uniform")
        self._uniform_btn.clicked.connect(self._on_uniform)
        self._clear_btn = QPushButton("Clear for this node")
        self._clear_btn.clicked.connect(self._on_clear_node)
        actions.addWidget(self._apply_btn)
        actions.addWidget(self._uniform_btn)
        actions.addWidget(self._clear_btn)
        actions.addStretch()
        layout.addLayout(actions)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.selection_changed.connect(self._on_selection)
        self._viewmodel.soft_evidence_changed.connect(lambda *_: self._refresh())
        self._viewmodel.structure_changed.connect(lambda *_: self._on_selection(self._current_node or ""))
        self._viewmodel.model_loaded.connect(lambda *_: self._reset())
        self._viewmodel.model_cleared.connect(self._reset)

    def _on_selection(self, node_id: str) -> None:
        self._current_node = node_id or None
        if not self._current_node:
            self._reset()
            return
        self._refresh()

    def _refresh(self) -> None:
        if not self._current_node:
            self._reset()
            return
        # The node may have been renamed / removed between the selection
        # event and this refresh — guard before touching the wrapper.
        try:
            states = self._viewmodel.model_service.get_outcomes(self._current_node)
        except Exception:
            self._current_node = None
            self._reset()
            return
        existing = self._viewmodel.soft_evidence.get(self._current_node, {})
        self._node_label.setText(f"Likelihoods for <b>{self._current_node}</b>")
        self._table.setRowCount(0)
        for state in states:
            row = self._table.rowCount()
            self._table.insertRow(row)
            state_item = QTableWidgetItem(state)
            state_item.setFlags(state_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, state_item)
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(0.0, 1e9)
            spin.setSingleStep(0.05)
            value = float(existing.get(state, 1.0))
            spin.setValue(value)
            self._table.setCellWidget(row, 1, spin)
        self._apply_btn.setEnabled(True)
        self._uniform_btn.setEnabled(True)
        self._clear_btn.setEnabled(bool(existing))

    def _reset(self) -> None:
        self._current_node = None
        self._table.setRowCount(0)
        self._node_label.setText("Select a node to edit its likelihood vector.")
        self._apply_btn.setEnabled(False)
        self._uniform_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)

    def _collect(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in range(self._table.rowCount()):
            state_item = self._table.item(row, 0)
            spin = self._table.cellWidget(row, 1)
            if state_item is None or spin is None:
                continue
            values[state_item.text()] = float(spin.value())
        return values

    def _on_apply(self) -> None:
        if not self._current_node:
            return
        values = self._collect()
        if sum(values.values()) <= 0:
            QMessageBox.warning(
                self,
                "Invalid soft evidence",
                "Total mass must be positive — at least one state needs likelihood > 0.",
            )
            return
        # Normalise to preserve relative ratios without forcing sum = 1.
        self._viewmodel.set_soft_evidence(self._current_node, values)

    def _on_uniform(self) -> None:
        for row in range(self._table.rowCount()):
            spin = self._table.cellWidget(row, 1)
            if isinstance(spin, QDoubleSpinBox):
                spin.setValue(1.0)

    def _on_clear_node(self) -> None:
        if self._current_node:
            self._viewmodel.set_soft_evidence(self._current_node, None)
