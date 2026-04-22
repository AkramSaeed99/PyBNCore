"""Equation node editor — define a deterministic functional node.

The user supplies a Python function `f(*parent_states) -> str`. The function
must return one of the node's declared states for every parent combination.
Equations are session-local (callables cannot be serialised), and the dialog
warns about that.
"""
from __future__ import annotations

from itertools import product
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

_DEFAULT_CODE = """def f(*parents):
    # Return one of this node's state names based on the parents' states.
    if not parents:
        return "true"
    return "true" if any(p == "true" for p in parents) else "false"
"""


class EquationNodeDialog(QDialog):
    def __init__(
        self,
        existing_nodes: list[str],
        node_service,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._node_service = node_service
        self._existing = set(existing_nodes)
        self.setWindowTitle("Add Equation Node")
        self.setModal(True)
        self.resize(720, 600)
        self._result: tuple[str, tuple[str, ...], tuple[str, ...], Callable, str] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Alert")
        header_row.addWidget(self._name_edit, stretch=1)
        layout.addLayout(header_row)

        note = QLabel(
            "Equation nodes are session-local: the Python function is not "
            "serialised to XDSL or the project file."
        )
        note.setStyleSheet("color: #9b6a00; font-style: italic;")
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = QSplitter(Qt.Horizontal, self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("<b>States</b> (double-click to edit)"))
        self._states_list = QListWidget()
        self._states_list.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        for s in ("true", "false"):
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self._states_list.addItem(item)
        left_layout.addWidget(self._states_list)
        state_btns = QHBoxLayout()
        add_state = QPushButton("Add")
        add_state.clicked.connect(self._add_state)
        rm_state = QPushButton("Remove")
        rm_state.clicked.connect(self._remove_state)
        state_btns.addWidget(add_state)
        state_btns.addWidget(rm_state)
        left_layout.addLayout(state_btns)

        left_layout.addWidget(QLabel("<b>Parents</b>"))
        self._parents_list = QListWidget()
        self._parents_list.setSelectionMode(QAbstractItemView.MultiSelection)
        try:
            available = [n.id for n in self._node_service.list_nodes() if n.states]
        except Exception:
            available = []
        for p in sorted(available):
            self._parents_list.addItem(p)
        left_layout.addWidget(self._parents_list)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(
            QLabel("<b>Python expression</b> — define <tt>f(*parents)</tt>")
        )
        self._code_edit = QPlainTextEdit()
        self._code_edit.setPlainText(_DEFAULT_CODE)
        self._code_edit.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 12px;")
        right_layout.addWidget(self._code_edit, stretch=1)

        actions = QHBoxLayout()
        self._validate_btn = QPushButton("Validate")
        self._validate_btn.clicked.connect(self._validate)
        actions.addWidget(self._validate_btn)
        actions.addStretch()
        right_layout.addLayout(actions)

        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        right_layout.addWidget(self._validation_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #8a1c1c;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ----------------------------------------------------------- helpers

    def _add_state(self) -> None:
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
        return [i.text() for i in self._parents_list.selectedItems()]

    def _compile_function(self) -> Callable | None:
        code = self._code_edit.toPlainText()
        namespace: dict = {}
        try:
            exec(compile(code, "<equation>", "exec"), namespace)
        except Exception as e:  # noqa: BLE001
            self._validation_label.setText(
                f"<span style='color: #8a1c1c;'>Compile error: {e}</span>"
            )
            return None
        fn = namespace.get("f")
        if not callable(fn):
            self._validation_label.setText(
                "<span style='color: #8a1c1c;'>Code must define a function named <tt>f</tt>.</span>"
            )
            return None
        return fn

    def _validate(self) -> bool:
        self._validation_label.setText("")
        fn = self._compile_function()
        if fn is None:
            return False
        states = [s for s in self._current_states() if s]
        if len(states) < 2:
            self._validation_label.setText(
                "<span style='color: #8a1c1c;'>Need ≥ 2 unique states.</span>"
            )
            return False
        parents = self._current_parents()
        parent_state_lists: list[list[str]] = []
        for p in parents:
            try:
                parent_state_lists.append(list(self._node_service.get_outcomes(p)))
            except Exception as e:  # noqa: BLE001
                self._validation_label.setText(
                    f"<span style='color: #8a1c1c;'>Cannot read states for '{p}': {e}</span>"
                )
                return False

        combos = list(product(*parent_state_lists)) if parent_state_lists else [()]
        bad: list[str] = []
        for combo in combos:
            try:
                result = fn(*combo)
            except Exception as e:  # noqa: BLE001
                self._validation_label.setText(
                    f"<span style='color: #8a1c1c;'>Evaluation failed for {combo}: {e}</span>"
                )
                return False
            if result not in states:
                bad.append(f"{combo} → {result!r}")
                if len(bad) > 5:
                    bad.append("…")
                    break
        if bad:
            self._validation_label.setText(
                "<span style='color: #8a1c1c;'>Function returned invalid states:<br>"
                + "<br>".join(bad)
                + "</span>"
            )
            return False
        self._validation_label.setText(
            f"<span style='color: #2a7a2a;'>OK — {len(combos)} parent combination(s) covered.</span>"
        )
        return True

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return self._error("Name is required.")
        if any(c.isspace() for c in name):
            return self._error("Name cannot contain whitespace.")
        if name in self._existing:
            return self._error(f"A node named '{name}' already exists.")
        if not self._validate():
            return
        fn = self._compile_function()
        if fn is None:
            return
        states = [s for s in self._current_states() if s]
        parents = self._current_parents()
        self._result = (
            name,
            tuple(states),
            tuple(parents),
            fn,
            self._code_edit.toPlainText(),
        )
        self.accept()

    def _error(self, msg: str) -> None:
        self._error_label.setText(msg)
        QMessageBox.warning(self, "Invalid input", msg)

    def result_data(self):
        return self._result
