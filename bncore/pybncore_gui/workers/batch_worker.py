from __future__ import annotations

from typing import Mapping, Sequence

from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.workers.base_worker import BaseWorker


class BatchQueryWorker(BaseWorker):
    def __init__(
        self,
        inference_service: InferenceService,
        query_nodes: Sequence[str],
        evidence_rows: Sequence[Mapping[str, str]],
    ) -> None:
        super().__init__()
        self._service = inference_service
        self._nodes = list(query_nodes)
        self._rows = [dict(r) for r in evidence_rows]

    def _execute(self) -> object:
        self.progress.emit(5, f"Running batch query over {len(self._rows)} rows…")
        result = self._service.batch_query(self._nodes, self._rows)
        self.progress.emit(100, "Batch query complete")
        return result
