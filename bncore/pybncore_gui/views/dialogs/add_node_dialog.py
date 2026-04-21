"""Dialog for creating a new discrete node."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AddNodeDialog(QDialog):
    """Collect a unique node name and ≥ 2 state names."""

    def __init__(
        self,
        existing_names: set[str],
        parent: QWidget | None = None,
        *,
        default_states: tuple[str, ...] = ("state1", "state2"),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Discrete Node")
        self.setModal(True)
        self.resize(420, 360)
        self._existing = {n.lower() for n in existing_names}
        self._build_ui(default_states)

    def _build_ui(self, default_states: tuple[str, ...]) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Smoker")
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        layout.addWidget(QLabel("<b>States</b> (double-click to edit)"))

        list_row = QHBoxLayout()
        self._states_list = QListWidget()
        self._states_list.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self._states_list.setSelectionMode(QAbstractItemView.SingleSelection)
        for s in default_states:
            self._append_state(s)

        btn_col = QVBoxLayout()
        self._add_state_btn = QPushButton("Add")
        self._add_state_btn.clicked.connect(lambda: self._append_state("new_state"))
        self._remove_state_btn = QPushButton("Remove")
        self._remove_state_btn.clicked.connect(self._remove_selected)
        btn_col.addWidget(self._add_state_btn)
        btn_col.addWidget(self._remove_state_btn)
        btn_col.addStretch()

        list_row.addWidget(self._states_list, stretch=1)
        list_row.addLayout(btn_col)
        layout.addLayout(list_row, stretch=1)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #8a1c1c;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _append_state(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self._states_list.addItem(item)

    def _remove_selected(self) -> None:
        row = self._states_list.currentRow()
        if row >= 0:
            self._states_list.takeItem(row)

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._error("Name is required.")
            return
        if any(c.isspace() for c in name):
            self._error("Name cannot contain whitespace.")
            return
        if name.lower() in self._existing:
            self._error(f"A node named '{name}' already exists.")
            return

        states = []
        for i in range(self._states_list.count()):
            states.append(self._states_list.item(i).text().strip())
        if len(states) < 2:
            self._error("At least two states required.")
            return
        if any(not s for s in states):
            self._error("State names cannot be empty.")
            return
        if len(set(states)) != len(states):
            self._error("State names must be unique.")
            return

        self._name = name
        self._states = tuple(states)
        self.accept()

    def _error(self, msg: str) -> None:
        self._error_label.setText(msg)
        QMessageBox.warning(self, "Invalid input", msg)

    def result_data(self) -> tuple[str, tuple[str, ...]]:
        return self._name, self._states
