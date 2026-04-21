"""Dialog for renaming a node."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


class RenameNodeDialog(QDialog):
    def __init__(
        self,
        current_name: str,
        existing_names: set[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Rename '{current_name}'")
        self.setModal(True)
        self._current = current_name
        self._existing = {n.lower() for n in existing_names} - {current_name.lower()}
        self._new_name: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._edit = QLineEdit(self._current)
        self._edit.selectAll()
        form.addRow("New name:", self._edit)
        layout.addLayout(form)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #8a1c1c;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        name = self._edit.text().strip()
        if not name:
            self._error("Name is required.")
            return
        if any(c.isspace() for c in name):
            self._error("Name cannot contain whitespace.")
            return
        if name.lower() in self._existing:
            self._error(f"A node named '{name}' already exists.")
            return
        self._new_name = name
        self.accept()

    def _error(self, msg: str) -> None:
        self._error_label.setText(msg)
        QMessageBox.warning(self, "Invalid input", msg)

    def new_name(self) -> str:
        return self._new_name or self._current
