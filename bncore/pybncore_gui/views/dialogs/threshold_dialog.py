"""Add a rare-event threshold to an existing continuous node."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


class ThresholdDialog(QDialog):
    def __init__(
        self,
        continuous_nodes: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Rare-Event Threshold")
        self.setModal(True)
        self._result: tuple[str, float] | None = None
        self._build_ui(continuous_nodes)

    def _build_ui(self, nodes: list[str]) -> None:
        layout = QVBoxLayout(self)

        if not nodes:
            layout.addWidget(
                QLabel(
                    "No continuous nodes available. Create one before "
                    "adding a threshold."
                )
            )
            buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)
            return

        form = QFormLayout()
        self._node_combo = QComboBox()
        for n in sorted(nodes):
            self._node_combo.addItem(n)
        form.addRow("Continuous node:", self._node_combo)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(-1e18, 1e18)
        self._threshold_spin.setDecimals(6)
        self._threshold_spin.setValue(0.0)
        form.addRow("Threshold value:", self._threshold_spin)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Thresholds seed the discretization grid so posterior "
                "density near rare-event regions is better resolved.",
                wordWrap=True,
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        node = self._node_combo.currentText()
        threshold = float(self._threshold_spin.value())
        if not node:
            QMessageBox.warning(self, "Threshold", "Select a continuous node.")
            return
        self._result = (node, threshold)
        self.accept()

    def result_data(self) -> tuple[str, float] | None:
        return self._result
