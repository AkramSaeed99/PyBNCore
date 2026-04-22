from __future__ import annotations

from typing import Sequence

from pybncore_gui.services.analysis_service import AnalysisService
from pybncore_gui.workers.base_worker import BaseWorker


class MonteCarloWorker(BaseWorker):
    def __init__(
        self,
        service: AnalysisService,
        query_nodes: Sequence[str],
        observed_nodes: Sequence[str],
        num_samples: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._query = list(query_nodes)
        self._observed = list(observed_nodes)
        self._n = num_samples
        self._seed = seed

    def _execute(self) -> object:
        self.progress.emit(5, "Preparing Monte Carlo run…")
        result = self._service.monte_carlo(
            self._query,
            self._observed,
            num_samples=self._n,
            seed=self._seed,
            progress=self.progress.emit,
        )
        self.progress.emit(100, "Monte Carlo complete")
        return result
