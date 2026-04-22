from __future__ import annotations

from typing import Mapping, Sequence

from PySide6.QtCore import Slot

from pybncore_gui.domain.errors import QueryError
from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.workers.base_worker import BaseWorker


class HybridQueryWorker(BaseWorker):
    def __init__(
        self,
        inference_service: InferenceService,
        query_nodes: Sequence[str],
        *,
        max_iters: int = 8,
        eps_entropy: float = 1e-4,
        eps_kl: float = 1e-4,
        evidence: Mapping[str, str] | None = None,
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
        continuous_evidence: Mapping[str, float] | None = None,
        continuous_likelihoods: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._service = inference_service
        self._query_nodes = list(query_nodes)
        self._max_iters = max_iters
        self._eps_entropy = eps_entropy
        self._eps_kl = eps_kl
        self._evidence = dict(evidence or {})
        self._soft = {k: dict(v) for k, v in (soft_evidence or {}).items()}
        self._continuous_evidence = dict(continuous_evidence or {})
        self._continuous_likelihoods = dict(continuous_likelihoods or {})
        self._cancelled: bool = False

    @Slot()
    def cancel(self) -> None:
        """Cooperative cancel — flip the flag; `_execute` discards the
        result when the (atomic) wrapper call finally returns."""
        self._cancelled = True

    def _execute(self) -> object:
        self.progress.emit(
            5,
            f"Running hybrid inference over {len(self._query_nodes)} node(s)…",
        )
        result = self._service.run_hybrid(
            self._query_nodes,
            max_iters=self._max_iters,
            eps_entropy=self._eps_entropy,
            eps_kl=self._eps_kl,
            evidence=self._evidence,
            soft_evidence=self._soft,
            continuous_evidence=self._continuous_evidence,
            continuous_likelihoods=self._continuous_likelihoods,
        )
        if self._cancelled:
            raise QueryError("Cancelled by user")
        self.progress.emit(100, "Hybrid query complete")
        return result
