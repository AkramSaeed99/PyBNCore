from __future__ import annotations

from typing import Sequence

from pybncore_gui.services.analysis_service import AnalysisService
from pybncore_gui.workers.base_worker import BaseWorker


class BenchmarkWorker(BaseWorker):
    def __init__(
        self,
        service: AnalysisService,
        query_nodes: Sequence[str],
        observed_nodes: Sequence[str],
        row_counts: Sequence[int],
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._query = list(query_nodes)
        self._observed = list(observed_nodes)
        self._row_counts = list(row_counts)
        self._seed = seed

    def _execute(self) -> object:
        self.progress.emit(5, "Running benchmark…")
        result = self._service.benchmark(
            self._query,
            self._observed,
            self._row_counts,
            seed=self._seed,
            progress=self.progress.emit,
        )
        self.progress.emit(100, "Benchmark complete")
        return result
