from __future__ import annotations

from typing import Mapping, Sequence

from pybncore_gui.services.analysis_service import AnalysisService
from pybncore_gui.workers.base_worker import BaseWorker


class SensitivityWorker(BaseWorker):
    def __init__(
        self,
        service: AnalysisService,
        query_node: str,
        query_state: str,
        n_top: int,
        epsilon: float,
        evidence: Mapping[str, str] | None = None,
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._query_node = query_node
        self._query_state = query_state
        self._n_top = n_top
        self._epsilon = epsilon
        self._evidence = dict(evidence or {})
        self._soft = {k: dict(v) for k, v in (soft_evidence or {}).items()}

    def _execute(self) -> object:
        self.progress.emit(10, "Computing sensitivity ranking…")
        report = self._service.sensitivity_ranking(
            self._query_node,
            self._query_state,
            n_top=self._n_top,
            epsilon=self._epsilon,
            evidence=self._evidence,
            soft_evidence=self._soft,
        )
        self.progress.emit(100, "Sensitivity complete")
        return report


class VOIWorker(BaseWorker):
    def __init__(
        self,
        service: AnalysisService,
        query_node: str,
        candidate_nodes: Sequence[str] | None = None,
        evidence: Mapping[str, str] | None = None,
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._query_node = query_node
        self._candidates = list(candidate_nodes) if candidate_nodes else None
        self._evidence = dict(evidence or {})
        self._soft = {k: dict(v) for k, v in (soft_evidence or {}).items()}

    def _execute(self) -> object:
        self.progress.emit(10, "Computing VOI…")
        report = self._service.value_of_information(
            self._query_node,
            self._candidates,
            evidence=self._evidence,
            soft_evidence=self._soft,
        )
        self.progress.emit(100, "VOI complete")
        return report
