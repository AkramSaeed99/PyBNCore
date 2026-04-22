from __future__ import annotations

from typing import Mapping

from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.workers.base_worker import BaseWorker


class MAPQueryWorker(BaseWorker):
    def __init__(
        self,
        inference_service: InferenceService,
        evidence: Mapping[str, str],
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self._service = inference_service
        self._evidence = dict(evidence)
        self._soft = {k: dict(v) for k, v in (soft_evidence or {}).items()}

    def _execute(self) -> object:
        self.progress.emit(10, "Running MAP query…")
        result = self._service.query_map(self._evidence, self._soft)
        self.progress.emit(100, "MAP query complete")
        return result
