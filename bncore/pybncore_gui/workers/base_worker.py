"""Base class for background tasks running on a QThread."""
from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from pybncore_gui.domain.errors import DomainError

logger = logging.getLogger(__name__)


class BaseWorker(QObject):
    """Subclass and implement `_execute`. Exactly one terminal signal fires.

    Signals
    -------
    finished(object)     payload is a typed DTO (or None)
    failed(str, str)     user_message, traceback
    progress(int, str)   0-100 percent, label
    """

    finished = Signal(object)
    failed = Signal(str, str)
    progress = Signal(int, str)

    @Slot()
    def run(self) -> None:
        try:
            result = self._execute()
        except DomainError as e:
            logger.warning("Worker reported domain error: %s", e.user_message)
            self.failed.emit(e.user_message, traceback.format_exc())
            return
        except Exception as e:  # noqa: BLE001 — last-resort barrier
            logger.exception("Worker crashed")
            self.failed.emit(f"Unexpected error: {e}", traceback.format_exc())
            return
        self.finished.emit(result)

    def _execute(self) -> object:
        raise NotImplementedError
