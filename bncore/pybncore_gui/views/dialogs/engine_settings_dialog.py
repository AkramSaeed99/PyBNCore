"""Engine settings — triangulation heuristic, loopy BP, and JT stats."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pybncore_gui.domain.results import JTStats
from pybncore_gui.domain.settings import TRIANGULATION_HEURISTICS, EngineSettings


class EngineSettingsDialog(QDialog):
    def __init__(
        self,
        settings: EngineSettings,
        jt_stats: JTStats | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Engine Settings")
        self.setModal(True)
        self._result: EngineSettings | None = None
        self._build_ui(settings, jt_stats)

    def _build_ui(self, settings: EngineSettings, stats: JTStats | None) -> None:
        layout = QVBoxLayout(self)

        # -- Compilation -------------------------------------------------
        compile_group = QGroupBox("Compilation")
        compile_form = QFormLayout(compile_group)

        self._triangulation_combo = QComboBox()
        for heuristic in TRIANGULATION_HEURISTICS:
            self._triangulation_combo.addItem(heuristic, userData=heuristic)
        idx = self._triangulation_combo.findData(settings.triangulation)
        if idx >= 0:
            self._triangulation_combo.setCurrentIndex(idx)
        compile_form.addRow("Triangulation heuristic:", self._triangulation_combo)

        layout.addWidget(compile_group)

        # -- JT stats ----------------------------------------------------
        stats_group = QGroupBox("Junction tree stats (current compile)")
        stats_form = QFormLayout(stats_group)
        if stats is None:
            stats_form.addRow(
                QLabel("Not available — compile the model to populate.")
            )
        else:
            stats_form.addRow("Treewidth:", QLabel(str(stats.treewidth)))
            stats_form.addRow("Cliques:", QLabel(str(stats.num_cliques)))
            stats_form.addRow("Max clique size:", QLabel(str(stats.max_clique_size)))
            stats_form.addRow(
                "Total table entries:", QLabel(str(stats.total_table_entries))
            )
        layout.addWidget(stats_group)

        # -- Loopy BP ----------------------------------------------------
        self._lbp_group = QGroupBox("Loopy belief propagation (approximate)")
        self._lbp_group.setCheckable(True)
        self._lbp_group.setChecked(settings.use_loopy_bp)
        lbp_form = QFormLayout(self._lbp_group)

        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(1, 10_000)
        self._iterations_spin.setValue(settings.loopy_iterations)
        lbp_form.addRow("Max iterations:", self._iterations_spin)

        self._damping_spin = QDoubleSpinBox()
        self._damping_spin.setRange(0.0, 0.99)
        self._damping_spin.setSingleStep(0.05)
        self._damping_spin.setDecimals(3)
        self._damping_spin.setValue(settings.loopy_damping)
        lbp_form.addRow("Damping:", self._damping_spin)

        self._tolerance_spin = QDoubleSpinBox()
        self._tolerance_spin.setRange(1e-12, 1.0)
        self._tolerance_spin.setDecimals(12)
        self._tolerance_spin.setSingleStep(1e-6)
        self._tolerance_spin.setValue(settings.loopy_tolerance)
        lbp_form.addRow("Tolerance:", self._tolerance_spin)

        layout.addWidget(self._lbp_group)

        note = QLabel(
            "Loopy BP is a fallback for dense networks where junction-tree inference is infeasible. "
            "Soft evidence and batch queries are not yet supported in loopy mode."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #4a5363; font-style: italic;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._result = EngineSettings(
            use_loopy_bp=self._lbp_group.isChecked(),
            loopy_iterations=int(self._iterations_spin.value()),
            loopy_damping=float(self._damping_spin.value()),
            loopy_tolerance=float(self._tolerance_spin.value()),
            triangulation=str(self._triangulation_combo.currentData()),
        ).validated()
        self.accept()

    def result_settings(self) -> EngineSettings | None:
        return self._result
