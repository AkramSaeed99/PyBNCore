"""Node inspector: name, states, CPT — editable in Phase 2."""
from __future__ import annotations

from itertools import product

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.node import NodeModel
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel


class _CPTTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._row_labels: list[str] = []
        self._col_labels: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, 0))
        self._editable: bool = False

    def set_data(
        self,
        matrix: np.ndarray,
        row_labels: list[str],
        col_labels: list[str],
        editable: bool = True,
    ) -> None:
        self.beginResetModel()
        self._matrix = np.asarray(matrix, dtype=np.float64).copy()
        self._row_labels = row_labels
        self._col_labels = col_labels
        self._editable = editable
        self.endResetModel()

    def current_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else self._matrix.shape[0]

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008
        if parent.isValid():
            return 0
        return self._matrix.shape[1] if self._matrix.ndim == 2 else 0

    def flags(self, index: QModelIndex):  # type: ignore[override]
        base = super().flags(index)
        if self._editable and index.isValid():
            return base | Qt.ItemIsEditable
        return base

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            value = float(self._matrix[index.row(), index.column()])
            return f"{value:.6g}" if role == Qt.EditRole else f"{value:.4f}"
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:  # type: ignore[override]
        if not index.isValid() or role != Qt.EditRole:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(v) or v < 0 or v > 1:
            return False
        self._matrix[index.row(), index.column()] = v
        self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
        return True

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self._col_labels):
            return self._col_labels[section]
        if orientation == Qt.Vertical and 0 <= section < len(self._row_labels):
            return self._row_labels[section]
        return None


class NodeInspectorPanel(QWidget):
    rename_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._current: NodeModel | None = None
        self._cpt_model = _CPTTableModel()
        self._build_ui()
        self._bind_viewmodel()
        self._show_empty()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QFormLayout()
        self._name_label = QLabel("—")
        self._kind_label = QLabel("—")
        self._parents_label = QLabel("—")
        self._states_label = QLabel("—")
        for lbl in (self._name_label, self._kind_label, self._parents_label, self._states_label):
            lbl.setWordWrap(True)
        header.addRow("Node:", self._name_label)
        header.addRow("Kind:", self._kind_label)
        header.addRow("Parents:", self._parents_label)
        header.addRow("States:", self._states_label)
        layout.addLayout(header)

        action_row = QHBoxLayout()
        self._rename_btn = QPushButton("Rename…")
        self._rename_btn.clicked.connect(self._on_rename)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)
        action_row.addWidget(self._rename_btn)
        action_row.addWidget(self._delete_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        layout.addWidget(QLabel("<b>Description</b>"))
        self._description_edit = QPlainTextEdit()
        self._description_edit.setPlaceholderText(
            "Human-readable description of this node (GUI-only, saved to sidecar + XDSL comment)."
        )
        self._description_edit.setFixedHeight(80)
        layout.addWidget(self._description_edit)
        desc_actions = QHBoxLayout()
        self._save_desc_btn = QPushButton("Save description")
        self._save_desc_btn.clicked.connect(self._on_save_description)
        desc_actions.addStretch()
        desc_actions.addWidget(self._save_desc_btn)
        layout.addLayout(desc_actions)

        self._cpt_title = QLabel("Conditional probability table")
        self._cpt_title.setStyleSheet("font-weight: 600; margin-top: 4px;")
        layout.addWidget(self._cpt_title)

        self._cpt_view = QTableView(self)
        self._cpt_view.setModel(self._cpt_model)
        self._cpt_view.setAlternatingRowColors(True)
        self._cpt_view.horizontalHeader().setStretchLastSection(True)
        self._cpt_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._cpt_view, stretch=1)

        cpt_actions = QHBoxLayout()
        self._apply_btn = QPushButton("Apply CPT")
        self._apply_btn.clicked.connect(self._on_apply_cpt)
        self._normalize_btn = QPushButton("Normalize Rows")
        self._normalize_btn.setToolTip("Scale each row so it sums to 1.")
        self._normalize_btn.clicked.connect(self._on_normalize)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.clicked.connect(self._on_revert)
        cpt_actions.addWidget(self._apply_btn)
        cpt_actions.addWidget(self._normalize_btn)
        cpt_actions.addWidget(self._revert_btn)
        cpt_actions.addStretch()
        layout.addLayout(cpt_actions)

        self._empty_label = QLabel("Select a node to inspect it.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #6a7387;")
        layout.addWidget(self._empty_label)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.selection_changed.connect(self._on_selection_changed)
        self._viewmodel.model_loaded.connect(lambda *_: self._show_empty())
        self._viewmodel.structure_changed.connect(self._on_structure_changed)
        self._viewmodel.model_cleared.connect(self._show_empty)
        self._viewmodel.descriptions_changed.connect(lambda *_: self._refresh_description())

    # -------------------------------------------------------- internal logic

    def _on_selection_changed(self, node_id: str) -> None:
        if not node_id:
            self._show_empty()
            return
        nodes = self._viewmodel.model_service.list_nodes()
        match = next((n for n in nodes if n.id == node_id), None)
        if match is None:
            self._show_empty()
            return
        self._current = match
        self._show_node(match)
        self._refresh_description()

    def _on_structure_changed(self, *_args) -> None:
        # If our node was removed / renamed, selection will get updated elsewhere.
        # If it still exists, refresh content (shape may have changed).
        if self._current is None:
            return
        nodes = self._viewmodel.model_service.list_nodes()
        match = next((n for n in nodes if n.id == self._current.id), None)
        if match is None:
            self._show_empty()
        else:
            self._current = match
            self._show_node(match)

    def _show_node(self, node: NodeModel) -> None:
        self._name_label.setText(node.id)
        self._kind_label.setText(node.kind.value)
        self._parents_label.setText(", ".join(node.parents) or "—")
        self._states_label.setText(", ".join(node.states) or "—")
        self._empty_label.setVisible(False)
        self._set_controls_visible(True)
        self._load_cpt(node)

    def _load_cpt(self, node: NodeModel) -> None:
        cpt = self._viewmodel.model_service.get_cpt_shaped(node.id)
        if cpt is None or not node.states:
            self._cpt_model.set_data(np.zeros((0, 0)), [], [], editable=False)
            self._cpt_title.setText("Conditional probability table — unavailable")
            return
        self._cpt_title.setText("Conditional probability table")
        col_labels = list(node.states)
        parent_states_lists: list[list[str]] = []
        valid_parents = True
        for p in node.parents:
            try:
                parent_states_lists.append(self._viewmodel.model_service.get_outcomes(p))
            except Exception:
                valid_parents = False
                break

        n_states = len(node.states)
        try:
            matrix = np.asarray(cpt).reshape(-1, n_states)
        except ValueError:
            matrix = np.asarray(cpt).reshape(1, -1)

        if not node.parents:
            row_labels = ["(marginal)"]
        elif valid_parents:
            combos = list(product(*parent_states_lists))
            if len(combos) == matrix.shape[0]:
                row_labels = [
                    ", ".join(f"{node.parents[i]}={combo[i]}" for i in range(len(combo)))
                    for combo in combos
                ]
            else:
                row_labels = [f"row {i}" for i in range(matrix.shape[0])]
        else:
            row_labels = [f"row {i}" for i in range(matrix.shape[0])]

        self._cpt_model.set_data(matrix, row_labels, col_labels, editable=True)
        self._cpt_view.resizeColumnsToContents()

    def _show_empty(self) -> None:
        self._current = None
        self._name_label.setText("—")
        self._kind_label.setText("—")
        self._parents_label.setText("—")
        self._states_label.setText("—")
        self._cpt_model.set_data(np.zeros((0, 0)), [], [], editable=False)
        self._set_controls_visible(False)
        self._empty_label.setVisible(True)

    def _set_controls_visible(self, visible: bool) -> None:
        self._cpt_title.setVisible(visible)
        self._cpt_view.setVisible(visible)
        self._rename_btn.setEnabled(visible)
        self._delete_btn.setEnabled(visible)
        self._apply_btn.setEnabled(visible)
        self._normalize_btn.setEnabled(visible)
        self._revert_btn.setEnabled(visible)
        self._description_edit.setEnabled(visible)
        self._save_desc_btn.setEnabled(visible)
        if not visible:
            self._description_edit.blockSignals(True)
            self._description_edit.setPlainText("")
            self._description_edit.blockSignals(False)

    def _refresh_description(self) -> None:
        if self._current is None:
            return
        desc = self._viewmodel.descriptions.get(self._current.id, "")
        self._description_edit.blockSignals(True)
        try:
            if self._description_edit.toPlainText() != desc:
                self._description_edit.setPlainText(desc)
        finally:
            self._description_edit.blockSignals(False)

    def _on_save_description(self) -> None:
        if self._current is None:
            return
        self._viewmodel.set_description(
            self._current.id, self._description_edit.toPlainText()
        )

    # ------------------------------------------------------------- handlers

    def _on_rename(self) -> None:
        if self._current is not None:
            self.rename_requested.emit(self._current.id)

    def _on_delete(self) -> None:
        if self._current is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete node",
            f"Delete node '{self._current.id}'? "
            "This will also remove all incident edges.",
        )
        if reply == QMessageBox.Yes:
            self.delete_requested.emit(self._current.id)

    def _on_apply_cpt(self) -> None:
        if self._current is None:
            return
        matrix = self._cpt_model.current_matrix()
        if matrix.size == 0:
            return
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            QMessageBox.warning(
                self,
                "CPT not normalized",
                "Each row must sum to 1.0. Click 'Normalize Rows' or fix manually.",
            )
            return
        self._viewmodel.set_cpt(self._current.id, matrix)

    def _on_normalize(self) -> None:
        matrix = self._cpt_model.current_matrix()
        if matrix.size == 0:
            return
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = matrix / row_sums
        self._cpt_model.set_data(
            normalized,
            [
                self._cpt_model.headerData(i, Qt.Vertical, Qt.DisplayRole) or f"row {i}"
                for i in range(normalized.shape[0])
            ],
            [
                self._cpt_model.headerData(i, Qt.Horizontal, Qt.DisplayRole) or f"col {i}"
                for i in range(normalized.shape[1])
            ],
            editable=True,
        )

    def _on_revert(self) -> None:
        if self._current is not None:
            self._load_cpt(self._current)
