"""QApplication bootstrap and dependency wiring."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pybncore_gui.domain.session import ModelSession
from pybncore_gui.services.authoring_service import AuthoringService
from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.services.io_service import IOService
from pybncore_gui.services.model_service import ModelService
from pybncore_gui.services.validation_service import ValidationService
from pybncore_gui.viewmodels.main_viewmodel import MainViewModel
from pybncore_gui.views.main_window import MainWindow


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("PyBNCore GUI")
    app.setOrganizationName("PyBNCore")

    session = ModelSession()
    io_service = IOService(session)
    model_service = ModelService(session)
    inference_service = InferenceService(session)
    authoring_service = AuthoringService(session)
    validation_service = ValidationService(session)
    viewmodel = MainViewModel(
        session,
        io_service,
        model_service,
        inference_service,
        authoring_service,
        validation_service,
    )

    window = MainWindow(viewmodel)
    window.show()

    # Auto-open an XDSL passed on the command line.
    args = argv if argv is not None else sys.argv
    if len(args) > 1:
        candidate = Path(args[1])
        if candidate.is_file():
            viewmodel.open_xdsl(str(candidate))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
