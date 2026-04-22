"""Continuous evidence editor — point values and likelihood callables."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_LIKELIHOOD_TEMPLATE = """def f(x):
    # Return λ(x) — a non-negative likelihood density for the observation.
    # Example: noisy Gaussian observation around `x = 0.5`.
    import math
    sigma = 0.1
    return math.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
"""


class ContinuousEvidencePanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._continuous_nodes: list[str] = []
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel("<b>Continuous evidence</b>"))

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Node:"))
        self._node_combo = QComboBox()
        self._node_combo.currentTextChanged.connect(self._on_node_changed)
        row1.addWidget(self._node_combo, stretch=1)
        row1.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Value (hard)", userData="value")
        self._mode_combo.addItem("Likelihood λ(x)", userData="likelihood")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row1.addWidget(self._mode_combo)
        layout.addLayout(row1)

        self._stack = QStackedWidget()

        value_page = QWidget()
        vp = QHBoxLayout(value_page)
        vp.addWidget(QLabel("Observed value:"))
        self._value_spin = QDoubleSpinBox()
        self._value_spin.setRange(-1e18, 1e18)
        self._value_spin.setDecimals(6)
        vp.addWidget(self._value_spin)
        self._apply_value_btn = QPushButton("Pin value")
        self._apply_value_btn.clicked.connect(self._on_apply_value)
        vp.addWidget(self._apply_value_btn)
        self._clear_value_btn = QPushButton("Clear")
        self._clear_value_btn.clicked.connect(self._on_clear_value)
        vp.addWidget(self._clear_value_btn)
        vp.addStretch()
        self._stack.addWidget(value_page)

        lik_page = QWidget()
        lp = QVBoxLayout(lik_page)
        lp.addWidget(QLabel("Define a likelihood density λ(x):"))
        self._likelihood_code = QPlainTextEdit(_LIKELIHOOD_TEMPLATE)
        self._likelihood_code.setStyleSheet(
            "font-family: Menlo, Consolas, monospace; font-size: 12px;"
        )
        lp.addWidget(self._likelihood_code, stretch=1)
        lik_actions = QHBoxLayout()
        self._apply_lik_btn = QPushButton("Apply likelihood")
        self._apply_lik_btn.clicked.connect(self._on_apply_likelihood)
        self._clear_lik_btn = QPushButton("Clear")
        self._clear_lik_btn.clicked.connect(self._on_clear_likelihood)
        lik_actions.addStretch()
        lik_actions.addWidget(self._apply_lik_btn)
        lik_actions.addWidget(self._clear_lik_btn)
        lp.addLayout(lik_actions)
        self._stack.addWidget(lik_page)

        layout.addWidget(self._stack, stretch=1)

        layout.addWidget(QLabel("<b>Current continuous evidence</b>"))
        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Node", "Assignment"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._clear_all_btn = QPushButton("Clear all continuous evidence")
        self._clear_all_btn.clicked.connect(self._viewmodel.clear_all_continuous_evidence)
        layout.addWidget(self._clear_all_btn, alignment=Qt.AlignRight)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.continuous_nodes_changed.connect(self._on_nodes_changed)
        self._viewmodel.continuous_evidence_changed.connect(self._refresh_table)

    # ----------------------------------------------------------- handlers

    def _on_nodes_changed(self, names: list[str]) -> None:
        self._continuous_nodes = list(names)
        current = self._node_combo.currentText()
        self._node_combo.blockSignals(True)
        self._node_combo.clear()
        for n in self._continuous_nodes:
            self._node_combo.addItem(n)
        idx = self._node_combo.findText(current)
        if idx >= 0:
            self._node_combo.setCurrentIndex(idx)
        self._node_combo.blockSignals(False)
        self._on_node_changed(self._node_combo.currentText())
        self._refresh_table(
            self._viewmodel.continuous_evidence,
            self._viewmodel.continuous_likelihoods,
        )

    def _on_node_changed(self, _name: str) -> None:
        pass

    def _on_mode_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)

    def _on_apply_value(self) -> None:
        node = self._node_combo.currentText()
        if not node:
            return
        self._viewmodel.set_continuous_value(node, float(self._value_spin.value()))

    def _on_clear_value(self) -> None:
        node = self._node_combo.currentText()
        if not node:
            return
        self._viewmodel.set_continuous_value(node, None)

    def _on_apply_likelihood(self) -> None:
        node = self._node_combo.currentText()
        if not node:
            return
        fn = self._compile_likelihood()
        if fn is None:
            return
        self._viewmodel.set_continuous_likelihood(node, fn)

    def _on_clear_likelihood(self) -> None:
        node = self._node_combo.currentText()
        if not node:
            return
        self._viewmodel.set_continuous_likelihood(node, None)

    def _compile_likelihood(self) -> Callable[[float], float] | None:
        ns: dict = {}
        try:
            exec(compile(self._likelihood_code.toPlainText(), "<likelihood>", "exec"), ns)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Compile error", str(e))
            return None
        fn = ns.get("f")
        if not callable(fn):
            QMessageBox.warning(self, "Likelihood", "Code must define f(x).")
            return None
        # Sample the function to make sure it returns finite non-negative floats.
        for x in (0.0, 1e-3, 1.0):
            try:
                val = float(fn(x))
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Likelihood", f"f({x}) raised {e!r}")
                return None
            if val != val or val < 0:  # NaN or negative
                QMessageBox.warning(
                    self,
                    "Likelihood",
                    f"f({x}) returned {val}; likelihood must be non-negative.",
                )
                return None
        return fn

    def _refresh_table(self, evidence: dict, likelihoods: dict | None = None) -> None:
        likelihoods = likelihoods if likelihoods is not None else self._viewmodel.continuous_likelihoods
        self._table.setRowCount(0)
        for name, value in sorted(evidence.items()):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(name))
            self._table.setItem(row, 1, QTableWidgetItem(f"value = {value:.6g}"))
        for name in sorted(likelihoods.keys()):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(name))
            self._table.setItem(row, 1, QTableWidgetItem("likelihood λ(x)"))
