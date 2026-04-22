"""Scenario manager — save / load / apply / diff named evidence sets."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.scenario import Scenario
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel


class ScenariosPanel(QWidget):
    def __init__(self, viewmodel: MainViewModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._scenarios: list[Scenario] = []
        self._build_ui()
        self._bind_viewmodel()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Scenarios</b>"))
        header.addStretch()
        self._save_btn = QPushButton("Save current…")
        self._save_btn.clicked.connect(self._on_save_current)
        header.addWidget(self._save_btn)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal, self)

        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget(self)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self._list)

        actions = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        self._rename_btn = QPushButton("Rename…")
        self._rename_btn.clicked.connect(self._on_rename)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._on_delete)
        for b in (self._apply_btn, self._rename_btn, self._delete_btn):
            b.setEnabled(False)
        actions.addWidget(self._apply_btn)
        actions.addWidget(self._rename_btn)
        actions.addWidget(self._delete_btn)
        list_layout.addLayout(actions)

        splitter.addWidget(list_container)

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setPlaceholderText("Select a scenario to preview its evidence.")
        splitter.addWidget(self._details)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, stretch=1)

    def _bind_viewmodel(self) -> None:
        self._viewmodel.scenarios_changed.connect(self._populate)
        self._viewmodel.model_loaded.connect(lambda *_: self._populate(self._viewmodel.scenarios))
        self._viewmodel.model_cleared.connect(lambda: self._populate([]))

    def _populate(self, scenarios: list[Scenario]) -> None:
        self._scenarios = list(scenarios)
        current = self._list.currentItem().text() if self._list.currentItem() else ""
        self._list.clear()
        for s in self._scenarios:
            self._list.addItem(s.name)
        if current:
            for i in range(self._list.count()):
                if self._list.item(i).text() == current:
                    self._list.setCurrentRow(i)
                    return
        self._details.clear()
        self._set_actions_enabled(False)

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._scenarios):
            self._details.clear()
            self._set_actions_enabled(False)
            return
        scenario = self._scenarios[row]
        lines = [f"<b>{scenario.name}</b>", ""]
        if scenario.evidence:
            lines.append("<u>Hard evidence</u>")
            for k, v in sorted(scenario.evidence.items()):
                lines.append(f"  {k} = {v}")
        else:
            lines.append("<i>No hard evidence</i>")
        lines.append("")
        if scenario.soft_evidence:
            lines.append("<u>Soft evidence</u>")
            for node, likelihoods in sorted(scenario.soft_evidence.items()):
                row_desc = ", ".join(f"{k}:{v:.4g}" for k, v in likelihoods.items())
                lines.append(f"  {node}: {row_desc}")
        if scenario.notes:
            lines.append("")
            lines.append("<u>Notes</u>")
            lines.append(scenario.notes)
        self._details.setHtml("<br>".join(lines))
        self._set_actions_enabled(True)

    def _set_actions_enabled(self, enabled: bool) -> None:
        self._apply_btn.setEnabled(enabled)
        self._rename_btn.setEnabled(enabled)
        self._delete_btn.setEnabled(enabled)

    def _current_scenario(self) -> Scenario | None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._scenarios):
            return None
        return self._scenarios[row]

    def _on_save_current(self) -> None:
        name, ok = QInputDialog.getText(self, "Save scenario", "Scenario name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(s.name == name for s in self._scenarios):
            reply = QMessageBox.question(
                self,
                "Overwrite?",
                f"A scenario named '{name}' already exists. Overwrite it?",
            )
            if reply != QMessageBox.Yes:
                return
        self._viewmodel.save_current_scenario(name)

    def _on_apply(self) -> None:
        scenario = self._current_scenario()
        if scenario is None:
            return
        self._viewmodel.apply_scenario(scenario.name)

    def _on_rename(self) -> None:
        scenario = self._current_scenario()
        if scenario is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename scenario", "New name:", text=scenario.name
        )
        if not ok or not new_name.strip() or new_name == scenario.name:
            return
        self._viewmodel.rename_scenario(scenario.name, new_name.strip())

    def _on_delete(self) -> None:
        scenario = self._current_scenario()
        if scenario is None:
            return
        reply = QMessageBox.question(
            self, "Delete scenario", f"Delete scenario '{scenario.name}'?"
        )
        if reply != QMessageBox.Yes:
            return
        self._viewmodel.delete_scenario(scenario.name)
