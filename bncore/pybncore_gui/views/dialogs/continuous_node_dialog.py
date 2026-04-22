"""Multi-distribution continuous / deterministic node wizard."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.continuous import ContinuousDistKind, ContinuousNodeSpec

_DEFAULT_CODE = """def f(*parents):
    # Deterministic transformation. Each positional argument is the
    # numeric value of the corresponding parent variable.
    return float(sum(parents))
"""


class ContinuousNodeDialog(QDialog):
    def __init__(
        self,
        existing_node_ids: set[str],
        parent_options: list[str],
        parent_widget: QWidget | None = None,
    ) -> None:
        super().__init__(parent_widget)
        self._existing = set(existing_node_ids)
        self._parent_options = list(parent_options)
        self.setWindowTitle("Add Continuous Node")
        self.setModal(True)
        self.resize(720, 640)
        self._result: ContinuousNodeSpec | None = None
        self._build_ui()

    # ----------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        identity = QGroupBox("Identity")
        form = QFormLayout(identity)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Peak_Ground_Acceleration")
        form.addRow("Name:", self._name_edit)
        self._kind_combo = QComboBox()
        for kind in ContinuousDistKind:
            self._kind_combo.addItem(kind.value, userData=kind)
        form.addRow("Distribution:", self._kind_combo)
        self._domain_lo = QDoubleSpinBox()
        self._domain_lo.setRange(-1e18, 1e18)
        self._domain_lo.setDecimals(6)
        self._domain_lo.setValue(0.0)
        self._domain_hi = QDoubleSpinBox()
        self._domain_hi.setRange(-1e18, 1e18)
        self._domain_hi.setDecimals(6)
        self._domain_hi.setValue(1.0)
        dom_row = QHBoxLayout()
        dom_row.addWidget(self._domain_lo)
        dom_row.addWidget(QLabel("to"))
        dom_row.addWidget(self._domain_hi)
        form.addRow("Domain:", self._wrap(dom_row))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(2, 256)
        self._bins_spin.setValue(8)
        form.addRow("Initial bins:", self._bins_spin)
        self._rare_event = QCheckBox("Rare-event mode")
        form.addRow(self._rare_event)
        layout.addWidget(identity)

        self._parents_group = QGroupBox("Parents (required only for deterministic)")
        pl = QVBoxLayout(self._parents_group)
        self._parents_list = QListWidget()
        self._parents_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._parents_list.setFixedHeight(90)
        for p in sorted(self._parent_options):
            self._parents_list.addItem(p)
        pl.addWidget(self._parents_list)
        layout.addWidget(self._parents_group)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_normal_tab(), "Normal")
        self._tabs.addTab(self._build_lognormal_tab(), "LogNormal")
        self._tabs.addTab(self._build_uniform_tab(), "Uniform")
        self._tabs.addTab(self._build_exponential_tab(), "Exponential")
        self._tabs.addTab(self._build_deterministic_tab(), "Deterministic")
        self._kind_combo.currentIndexChanged.connect(
            lambda idx: self._tabs.setCurrentIndex(idx)
        )
        self._tabs.currentChanged.connect(
            lambda idx: self._kind_combo.setCurrentIndex(idx)
        )
        layout.addWidget(self._tabs, stretch=1)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #8a1c1c;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _wrap(layout: QHBoxLayout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _build_normal_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._mu = QDoubleSpinBox()
        self._mu.setRange(-1e12, 1e12)
        self._mu.setDecimals(6)
        self._mu.setValue(0.0)
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(1e-9, 1e12)
        self._sigma.setDecimals(6)
        self._sigma.setValue(1.0)
        form.addRow("Mean μ:", self._mu)
        form.addRow("Std σ:", self._sigma)
        return w

    def _build_lognormal_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._log_mu = QDoubleSpinBox()
        self._log_mu.setRange(-1e12, 1e12)
        self._log_mu.setDecimals(6)
        self._log_mu.setValue(0.0)
        self._log_sigma = QDoubleSpinBox()
        self._log_sigma.setRange(1e-9, 1e12)
        self._log_sigma.setDecimals(6)
        self._log_sigma.setValue(1.0)
        self._log_log_spaced = QCheckBox("Log-spaced bins")
        self._log_log_spaced.setChecked(True)
        form.addRow("log μ:", self._log_mu)
        form.addRow("log σ:", self._log_sigma)
        form.addRow(self._log_log_spaced)
        return w

    def _build_uniform_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._u_a = QDoubleSpinBox()
        self._u_a.setRange(-1e12, 1e12)
        self._u_a.setDecimals(6)
        self._u_a.setValue(0.0)
        self._u_b = QDoubleSpinBox()
        self._u_b.setRange(-1e12, 1e12)
        self._u_b.setDecimals(6)
        self._u_b.setValue(1.0)
        form.addRow("a:", self._u_a)
        form.addRow("b:", self._u_b)
        return w

    def _build_exponential_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._exp_rate = QDoubleSpinBox()
        self._exp_rate.setRange(1e-12, 1e12)
        self._exp_rate.setDecimals(6)
        self._exp_rate.setValue(1.0)
        self._exp_log_spaced = QCheckBox("Log-spaced bins")
        self._exp_log_spaced.setChecked(True)
        form.addRow("Rate λ:", self._exp_rate)
        form.addRow(self._exp_log_spaced)
        return w

    def _build_deterministic_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(
            QLabel(
                "Define <tt>f(*parents)</tt> returning a float. Parents are "
                "passed as positional floats in the order they appear above."
            )
        )
        self._det_code = QPlainTextEdit(_DEFAULT_CODE)
        self._det_code.setStyleSheet(
            "font-family: Menlo, Consolas, monospace; font-size: 12px;"
        )
        layout.addWidget(self._det_code, stretch=1)
        form = QFormLayout()
        self._det_monotone = QCheckBox("Monotone in all parents")
        form.addRow(self._det_monotone)
        self._det_n_samples = QSpinBox()
        self._det_n_samples.setRange(4, 1024)
        self._det_n_samples.setValue(32)
        form.addRow("Sampling resolution:", self._det_n_samples)
        layout.addLayout(form)
        return w

    # ------------------------------------------------------------- accept

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return self._err("Name is required.")
        if any(c.isspace() for c in name):
            return self._err("Name cannot contain whitespace.")
        if name in self._existing:
            return self._err(f"A node named '{name}' already exists.")
        lo = float(self._domain_lo.value())
        hi = float(self._domain_hi.value())
        if not (hi > lo):
            return self._err("Domain upper bound must exceed lower bound.")
        kind: ContinuousDistKind = self._kind_combo.currentData()
        parents = tuple(i.text() for i in self._parents_list.selectedItems())

        params: dict = {}
        fn: Callable | None = None
        log_spaced = False
        monotone = False
        n_samples = 32

        if kind is ContinuousDistKind.NORMAL:
            params = {"mu": float(self._mu.value()), "sigma": float(self._sigma.value())}
        elif kind is ContinuousDistKind.LOGNORMAL:
            params = {
                "log_mu": float(self._log_mu.value()),
                "log_sigma": float(self._log_sigma.value()),
            }
            log_spaced = bool(self._log_log_spaced.isChecked())
        elif kind is ContinuousDistKind.UNIFORM:
            params = {"a": float(self._u_a.value()), "b": float(self._u_b.value())}
            if params["b"] <= params["a"]:
                return self._err("Uniform b must exceed a.")
        elif kind is ContinuousDistKind.EXPONENTIAL:
            params = {"rate": float(self._exp_rate.value())}
            log_spaced = bool(self._exp_log_spaced.isChecked())
        elif kind is ContinuousDistKind.DETERMINISTIC:
            if not parents:
                return self._err("Deterministic nodes need at least one parent.")
            fn = self._compile_function()
            if fn is None:
                return
            monotone = bool(self._det_monotone.isChecked())
            n_samples = int(self._det_n_samples.value())

        self._result = ContinuousNodeSpec(
            name=name,
            kind=kind,
            parents=parents,
            domain=(lo, hi),
            initial_bins=int(self._bins_spin.value()),
            rare_event_mode=bool(self._rare_event.isChecked()),
            log_spaced=log_spaced,
            monotone=monotone,
            n_samples=n_samples,
            params=params,
            fn=fn,
        )
        self.accept()

    def _compile_function(self) -> Callable | None:
        namespace: dict = {}
        try:
            exec(compile(self._det_code.toPlainText(), "<deterministic>", "exec"), namespace)
        except Exception as e:  # noqa: BLE001
            self._err(f"Compile error: {e}")
            return None
        fn = namespace.get("f")
        if not callable(fn):
            self._err("Code must define a function named 'f'.")
            return None
        return fn

    def _err(self, msg: str) -> None:
        self._error_label.setText(msg)
        QMessageBox.warning(self, "Invalid input", msg)

    def result_spec(self) -> ContinuousNodeSpec | None:
        return self._result
