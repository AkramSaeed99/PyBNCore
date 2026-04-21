"""Shows a ValidationReport — one row per issue, color-coded by severity."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.validation import Severity, ValidationReport
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_COLOR = {
    Severity.ERROR: QColor("#fde2e2"),
    Severity.WARNING: QColor("#fff4d6"),
    Severity.INFO: QColor("#eaf3ff"),
}


class ValidationPanel(QWidget):
    node_activated = Signal(str)

    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Validation</b>"))
        header.addStretch()
        self._revalidate_btn = QPushButton("Run Validation")
        self._revalidate_btn.clicked.connect(self._viewmodel.validate)
        header.addWidget(self._revalidate_btn)
        layout.addLayout(header)

        self._summary = QLabel("No validation run yet.")
        self._summary.setStyleSheet("color: #4a5363;")
        layout.addWidget(self._summary)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Severity", "Code", "Node", "Message"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.itemDoubleClicked.connect(self._on_activated)
        layout.addWidget(self._table, stretch=1)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.validation_report.connect(self._display)
        self._viewmodel.structure_changed.connect(lambda *_: self._reset())
        self._viewmodel.model_loaded.connect(lambda *_: self._reset())
        self._viewmodel.model_cleared.connect(self._reset)

    def _display(self, report: ValidationReport) -> None:
        self._table.setRowCount(0)
        issues = report.issues
        if not issues:
            self._summary.setText("No issues detected.")
            return
        errors = sum(1 for i in issues if i.severity == Severity.ERROR)
        warnings = sum(1 for i in issues if i.severity == Severity.WARNING)
        infos = sum(1 for i in issues if i.severity == Severity.INFO)
        self._summary.setText(
            f"{len(issues)} issue(s) — {errors} errors, {warnings} warnings, {infos} info."
        )
        for issue in issues:
            row = self._table.rowCount()
            self._table.insertRow(row)
            items = [
                QTableWidgetItem(issue.severity.value.upper()),
                QTableWidgetItem(issue.code),
                QTableWidgetItem(issue.node or ""),
                QTableWidgetItem(issue.message),
            ]
            for item in items:
                item.setBackground(_COLOR.get(issue.severity, QColor("#ffffff")))
            for col, item in enumerate(items):
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

    def _reset(self) -> None:
        self._table.setRowCount(0)
        self._summary.setText("No validation run yet.")

    def _on_activated(self, item: QTableWidgetItem) -> None:
        row = item.row()
        node_item = self._table.item(row, 2)
        if node_item and node_item.text():
            self.node_activated.emit(node_item.text())
            self._viewmodel.select_node(node_item.text())
