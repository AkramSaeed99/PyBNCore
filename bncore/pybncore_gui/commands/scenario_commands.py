"""QUndoCommand subclasses for scenario CRUD.

Scenarios are pure viewmodel state — no wrapper calls. Each command
directly mutates the underlying list and notifies the viewmodel so the
UI picks up the change.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QUndoCommand

from pybncore_gui.domain.scenario import Scenario

NotifyCallback = Callable[[], None]


class _BaseScenarioCommand(QUndoCommand):
    def __init__(self, text: str, scenarios: list[Scenario], notify: NotifyCallback) -> None:
        super().__init__(text)
        self._scenarios = scenarios
        self._notify = notify


class SaveScenarioCommand(_BaseScenarioCommand):
    """Insert or replace a named scenario."""

    def __init__(
        self,
        scenarios: list[Scenario],
        notify: NotifyCallback,
        scenario: Scenario,
    ) -> None:
        super().__init__(f"Save scenario '{scenario.name}'", scenarios, notify)
        self._scenario = scenario
        self._prior_index: int | None = None
        self._prior: Scenario | None = None

    def redo(self) -> None:  # type: ignore[override]
        for i, existing in enumerate(self._scenarios):
            if existing.name == self._scenario.name:
                self._prior_index = i
                self._prior = existing
                self._scenarios[i] = self._scenario
                break
        else:
            self._prior_index = len(self._scenarios)
            self._prior = None
            self._scenarios.append(self._scenario)
        self._notify()

    def undo(self) -> None:  # type: ignore[override]
        if self._prior_index is None:
            return
        if self._prior is not None:
            self._scenarios[self._prior_index] = self._prior
        else:
            # Was newly appended — pop it.
            if 0 <= self._prior_index < len(self._scenarios):
                self._scenarios.pop(self._prior_index)
        self._notify()


class DeleteScenarioCommand(_BaseScenarioCommand):
    def __init__(
        self,
        scenarios: list[Scenario],
        notify: NotifyCallback,
        name: str,
    ) -> None:
        super().__init__(f"Delete scenario '{name}'", scenarios, notify)
        self._name = name
        self._prior_index: int | None = None
        self._prior: Scenario | None = None

    def redo(self) -> None:  # type: ignore[override]
        for i, existing in enumerate(self._scenarios):
            if existing.name == self._name:
                self._prior_index = i
                self._prior = existing
                self._scenarios.pop(i)
                self._notify()
                return

    def undo(self) -> None:  # type: ignore[override]
        if self._prior is None or self._prior_index is None:
            return
        self._scenarios.insert(self._prior_index, self._prior)
        self._notify()


class RenameScenarioCommand(_BaseScenarioCommand):
    def __init__(
        self,
        scenarios: list[Scenario],
        notify: NotifyCallback,
        old: str,
        new: str,
    ) -> None:
        super().__init__(f"Rename scenario '{old}' → '{new}'", scenarios, notify)
        self._old = old
        self._new = new

    def redo(self) -> None:  # type: ignore[override]
        for s in self._scenarios:
            if s.name == self._old:
                s.name = self._new
                self._notify()
                return

    def undo(self) -> None:  # type: ignore[override]
        for s in self._scenarios:
            if s.name == self._new:
                s.name = self._old
                self._notify()
                return
