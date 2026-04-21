"""Evidence editor: one combo box per node, synced to the viewmodel."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.node import EdgeModel, NodeModel
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel

_NONE_LABEL = "(none)"


class EvidencePanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._combos: dict[str, QComboBox] = {}
        self._suppress_events = False
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Hard evidence</b>"))
        header.addStretch()
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.clicked.connect(self._viewmodel.clear_evidence)
        header.addWidget(self._clear_btn)
        outer.addLayout(header)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._form = QFormLayout(self._container)
        self._form.setContentsMargins(4, 4, 4, 4)
        self._form.setSpacing(4)
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll, stretch=1)

        self._empty_label = QLabel("Load a model to edit evidence.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #6a7387;")
        outer.addWidget(self._empty_label)
        self._scroll.setVisible(False)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.model_loaded.connect(self._populate)
        self._viewmodel.evidence_changed.connect(self._reflect_evidence)
        self._viewmodel.model_cleared.connect(self._reset)

    def _populate(self, nodes: list[NodeModel], _edges: list[EdgeModel]) -> None:
        self._reset()
        usable = [n for n in nodes if n.states]
        if not usable:
            self._empty_label.setText("No discrete nodes available for evidence.")
            return
        for node in sorted(usable, key=lambda n: n.id):
            combo = QComboBox(self._container)
            combo.addItem(_NONE_LABEL, userData=None)
            for state in node.states:
                combo.addItem(state, userData=state)
            combo.currentIndexChanged.connect(
                lambda _idx, nid=node.id, c=combo: self._on_combo_changed(nid, c)
            )
            self._combos[node.id] = combo
            self._form.addRow(node.id, combo)
        self._empty_label.setVisible(False)
        self._scroll.setVisible(True)

    def _reset(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._combos.clear()
        self._empty_label.setText("Load a model to edit evidence.")
        self._empty_label.setVisible(True)
        self._scroll.setVisible(False)

    def _on_combo_changed(self, node_id: str, combo: QComboBox) -> None:
        if self._suppress_events:
            return
        state = combo.currentData()
        self._viewmodel.set_evidence(node_id, state)

    def _reflect_evidence(self, evidence: dict) -> None:
        self._suppress_events = True
        try:
            for node_id, combo in self._combos.items():
                state = evidence.get(node_id)
                if state is None:
                    combo.setCurrentIndex(0)
                    continue
                idx = combo.findData(state)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._suppress_events = False
