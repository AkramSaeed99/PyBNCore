from __future__ import annotations

from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.workers.base_worker import BaseWorker


class CompileWorker(BaseWorker):
    def __init__(self, inference_service: InferenceService) -> None:
        super().__init__()
        self._service = inference_service

    def _execute(self) -> object:
        self.progress.emit(10, "Compiling junction tree…")
        stats = self._service.compile()
        self.progress.emit(100, "Compile complete")
        return stats
