"""Rewrite the discrete state list of an existing node.

Changing states resets the node's CPT and every child's CPT because the
factor shapes depend on the new cardinalities. The dialog surfaces that
explicitly so the user doesn't silently lose CPT content.
"""
from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EditStatesDialog(QDialog):
    def __init__(
        self,
        node_id: str,
        current_states: Sequence[str],
        affected_children: Sequence[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit States — {node_id}")
        self.setModal(True)
        self.resize(460, 420)
        self._node_id = node_id
        self._affected_children = list(affected_children)
        self._result: tuple[str, ...] | None = None
        self._build_ui(list(current_states))

    def _build_ui(self, states: list[str]) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(
                f"<b>{self._node_id}</b> — double-click to edit, drag to reorder."
            )
        )

        self._list = QListWidget()
        self._list.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        for s in states:
            self._append(s)
        layout.addWidget(self._list, stretch=1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add state")
        add_btn.clicked.connect(lambda: self._append("new_state"))
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        warning_parts = [
            "<b>Note:</b> changing states resets this node's CPT to uniform."
        ]
        if self._affected_children:
            joined = ", ".join(self._affected_children)
            warning_parts.append(
                f"The following children also have their CPTs reset: <i>{joined}</i>."
            )
        warning = QLabel(" ".join(warning_parts))
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #9b6a00;")
        layout.addWidget(warning)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #8a1c1c;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self._list.addItem(item)

    def _remove_selected(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)

    def _on_accept(self) -> None:
        raw = [
            self._list.item(i).text().strip() for i in range(self._list.count())
        ]
        if len(raw) < 2:
            return self._err("At least two states are required.")
        if any(not s for s in raw):
            return self._err("State names cannot be empty.")
        if len(set(raw)) != len(raw):
            return self._err("State names must be unique.")
        self._result = tuple(raw)
        self.accept()

    def _err(self, msg: str) -> None:
        self._error_label.setText(msg)
        QMessageBox.warning(self, "Invalid states", msg)

    def result_states(self) -> tuple[str, ...] | None:
        return self._result
