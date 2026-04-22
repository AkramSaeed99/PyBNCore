"""Noisy-MAX wizard — create a new noisy-max node from parent link matrices."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

_DEFAULT_STATES = ("none", "low", "high")


class NoisyMaxWizard(QDialog):
    def __init__(
        self,
        existing_nodes: list[str],
        node_service,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._node_service = node_service
        self._existing = set(existing_nodes)
        self.setWindowTitle("Add Noisy-MAX Node")
        self.setModal(True)
        self.resize(720, 620)
        self._result: tuple[str, tuple[str, ...], tuple[str, ...], dict[str, np.ndarray], np.ndarray] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --------------------- Header: name + states + parents
        top = QGroupBox("Identity")
        top_layout = QFormLayout(top)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Disease")
        top_layout.addRow("Name:", self._name_edit)

        self._states_list = QListWidget()
        self._states_list.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        for s in _DEFAULT_STATES:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self._states_list.addItem(item)
        states_row = QHBoxLayout()
        states_row.addWidget(self._states_list)
        state_btns = QVBoxLayout()
        add_state_btn = QPushButton("Add")
        add_state_btn.clicked.connect(lambda: self._append_state())
        rm_state_btn = QPushButton("Remove")
        rm_state_btn.clicked.connect(self._remove_state)
        state_btns.addWidget(add_state_btn)
        state_btns.addWidget(rm_state_btn)
        state_btns.addStretch()
        states_row.addLayout(state_btns)
        top_layout.addRow("States (ordered low → high):", self._pack(states_row))
        layout.addWidget(top)

        # --------------------- Parents picker
        parents_group = QGroupBox("Parents")
        parents_layout = QVBoxLayout(parents_group)
        self._parent_list = QListWidget()
        self._parent_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._parent_list.setFixedHeight(110)
        try:
            available = [n.id for n in self._node_service.list_nodes() if n.states]
        except Exception:
            available = []
        for p in sorted(available):
            self._parent_list.addItem(p)
        parents_layout.addWidget(self._parent_list)
        rebuild_btn = QPushButton("Rebuild matrices")
        rebuild_btn.clicked.connect(self._rebuild_matrices)
        parents_layout.addWidget(rebuild_btn, alignment=Qt.AlignRight)
        layout.addWidget(parents_group)

        # --------------------- Matrix editors + leak
        splitter = QSplitter(Qt.Vertical, self)

        self._matrix_tabs = QTabWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._matrix_tabs)
        splitter.addWidget(self._wrap_group("Link matrices (rows=parent state, cols=child state)", scroll))

        self._leak_table = QTableWidget(0, 0, self)
        self._leak_table.verticalHeader().setVisible(False)
        self._leak_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        splitter.addWidget(self._wrap_group("Leak probabilities (sum to 1)", self._leak_table))
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #8a1c1c;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _pack(sublayout):
        w = QWidget()
        w.setLayout(sublayout)
        return w

    @staticmethod
    def _wrap_group(title: str, inner: QWidget) -> QWidget:
        group = QGroupBox(title)
        lay = QVBoxLayout(group)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(inner)
        return group

    # ------------------------------------------------------ state list ops

    def _append_state(self) -> None:
        item = QListWidgetItem("state")
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self._states_list.addItem(item)

    def _remove_state(self) -> None:
        row = self._states_list.currentRow()
        if row >= 0:
            self._states_list.takeItem(row)

    def _current_states(self) -> list[str]:
        return [self._states_list.item(i).text().strip() for i in range(self._states_list.count())]

    def _current_parents(self) -> list[str]:
        return [i.text() for i in self._parent_list.selectedItems()]

    # --------------------------------------------------- matrix tabs

    def _rebuild_matrices(self) -> None:
        self._matrix_tabs.clear()
        states = [s for s in self._current_states() if s]
        parents = self._current_parents()
        if not states or not parents:
            return

        for parent in parents:
            try:
                parent_states = self._node_service.get_outcomes(parent)
            except Exception:
                parent_states = []
            table = QTableWidget(len(parent_states), len(states), self._matrix_tabs)
            table.setVerticalHeaderLabels(list(parent_states))
            table.setHorizontalHeaderLabels(list(states))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            # Default: identity-ish — max-effect state for highest-indexed parent state, etc.
            for r in range(len(parent_states)):
                for c in range(len(states)):
                    if len(parent_states) == len(states) and r == c:
                        v = 1.0
                    else:
                        v = 1.0 / max(1, len(states))
                    item = QTableWidgetItem(f"{v:.4f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(r, c, item)
            self._matrix_tabs.addTab(table, parent)

        # Leak vector
        self._leak_table.clear()
        self._leak_table.setRowCount(1)
        self._leak_table.setColumnCount(len(states))
        self._leak_table.setHorizontalHeaderLabels(states)
        self._leak_table.setVerticalHeaderLabels(["leak"])
        # Default: put all mass on first (= "none") state.
        for c, _s in enumerate(states):
            val = 1.0 if c == 0 else 0.0
            item = QTableWidgetItem(f"{val:.4f}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._leak_table.setItem(0, c, item)
        self._leak_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # --------------------------------------------------- accept

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return self._error("Name is required.")
        if any(c.isspace() for c in name):
            return self._error("Name cannot contain whitespace.")
        if name in self._existing:
            return self._error(f"A node named '{name}' already exists.")
        states = [s for s in self._current_states() if s]
        if len(states) < 2 or len(set(states)) != len(states):
            return self._error("Provide ≥ 2 unique state names.")
        parents = self._current_parents()
        if not parents:
            return self._error("Select at least one parent.")

        link_matrices: dict[str, np.ndarray] = {}
        for i in range(self._matrix_tabs.count()):
            widget = self._matrix_tabs.widget(i)
            parent = self._matrix_tabs.tabText(i)
            if not isinstance(widget, QTableWidget):
                continue
            m = np.zeros((widget.rowCount(), widget.columnCount()), dtype=np.float64)
            for r in range(widget.rowCount()):
                for c in range(widget.columnCount()):
                    item = widget.item(r, c)
                    try:
                        m[r, c] = float(item.text()) if item and item.text() else 0.0
                    except ValueError:
                        return self._error(
                            f"Invalid number at {parent}[{r}, {c}]."
                        )
            link_matrices[parent] = m

        if set(link_matrices.keys()) != set(parents):
            return self._error(
                "Link matrices missing for one or more parents. Click 'Rebuild matrices'."
            )

        leak = np.zeros(len(states), dtype=np.float64)
        for c in range(self._leak_table.columnCount()):
            item = self._leak_table.item(0, c)
            try:
                leak[c] = float(item.text()) if item and item.text() else 0.0
            except ValueError:
                return self._error(f"Invalid leak value at column {c}.")
        if not np.isclose(leak.sum(), 1.0, atol=1e-6):
            return self._error("Leak probabilities must sum to 1.0.")

        self._result = (name, tuple(states), tuple(parents), link_matrices, leak)
        self.accept()

    def _error(self, msg: str) -> None:
        self._error_label.setText(msg)
        QMessageBox.warning(self, "Invalid input", msg)

    def result_data(self):
        return self._result
