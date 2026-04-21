from __future__ import annotations

from typing import Mapping

from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.workers.base_worker import BaseWorker


class QueryWorker(BaseWorker):
    def __init__(
        self,
        inference_service: InferenceService,
        node: str,
        evidence: Mapping[str, str],
    ) -> None:
        super().__init__()
        self._service = inference_service
        self._node = node
        self._evidence = dict(evidence)

    def _execute(self) -> object:
        self.progress.emit(10, f"Querying '{self._node}'…")
        result = self._service.query_single(self._node, self._evidence)
        self.progress.emit(100, "Query complete")
        return result
