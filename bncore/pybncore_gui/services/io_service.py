"""Load and save models (XDSL read/write, BIF import, project sidecars)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
from pybncore import PyBNCoreWrapper
from pybncore.io import read_bif, write_xdsl

from pybncore_gui.domain.errors import ModelIOError
from pybncore_gui.domain.project import ProjectFile
from pybncore_gui.domain.session import ModelSession

logger = logging.getLogger(__name__)


class IOService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session

    # --------------------------------------------------------- XDSL / BIF IO

    def open_xdsl(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise ModelIOError(f"File not found: {path}")
        try:
            wrapper = PyBNCoreWrapper.from_xdsl(str(path))
        except Exception as e:  # noqa: BLE001
            logger.exception("XDSL load failed: %s", path)
            raise ModelIOError(f"Failed to open '{path.name}': {e}") from e
        self._session.set_wrapper(wrapper, source_path=str(path))

    def import_bif(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise ModelIOError(f"File not found: {path}")
        try:
            graph, cpts = read_bif(str(path))
        except Exception as e:  # noqa: BLE001
            logger.exception("BIF read failed: %s", path)
            raise ModelIOError(f"Failed to import '{path.name}': {e}") from e
        try:
            wrapper = PyBNCoreWrapper()
            wrapper._graph = graph
            wrapper._cpts = {k: np.asarray(v, dtype=np.float64) for k, v in cpts.items()}
            wrapper._cache_metadata()
        except Exception as e:  # noqa: BLE001
            raise ModelIOError(f"Unable to initialise model from BIF: {e}") from e
        self._session.set_wrapper(wrapper, source_path=None)

    def save_xdsl(self, path: str | Path) -> None:
        path = Path(path)
        with self._session.locked() as wrapper:
            if wrapper is None or wrapper._graph is None:
                raise ModelIOError("No model to save.")
            try:
                write_xdsl(wrapper._graph, dict(wrapper._cpts), str(path))
            except Exception as e:  # noqa: BLE001
                logger.exception("XDSL save failed: %s", path)
                raise ModelIOError(f"Failed to save '{path.name}': {e}") from e
        self._session.set_source_path(str(path))

    def new_empty(self) -> None:
        try:
            wrapper = PyBNCoreWrapper()
        except Exception as e:  # noqa: BLE001
            raise ModelIOError(f"Unable to initialise empty model: {e}") from e
        self._session.set_wrapper(wrapper, source_path=None)

    # ------------------------------------------------------------ projects

    def save_project(
        self,
        path: str | Path,
        positions: Mapping[str, tuple[float, float]],
        scenarios: list[dict] | None = None,
    ) -> Path:
        """Write `<path>.pbnproj` sidecar alongside the current XDSL.

        The XDSL itself is saved next to it with the same basename if the
        model does not yet have an on-disk source.
        """
        path = Path(path)
        project_dir = path.parent
        project_dir.mkdir(parents=True, exist_ok=True)

        xdsl_path = path.with_suffix(".xdsl")
        self.save_xdsl(xdsl_path)

        project = ProjectFile(
            xdsl_relative=xdsl_path.name,
            positions=dict(positions),
            scenarios=list(scenarios or []),
        )
        project_path = path.with_suffix(".pbnproj")
        try:
            project.write(project_path)
        except OSError as e:
            raise ModelIOError(f"Unable to write project file: {e}") from e
        return project_path

    def load_project(self, path: str | Path) -> ProjectFile:
        path = Path(path)
        if not path.is_file():
            raise ModelIOError(f"Project file not found: {path}")
        try:
            project = ProjectFile.read(path)
        except Exception as e:  # noqa: BLE001
            raise ModelIOError(f"Invalid project file '{path.name}': {e}") from e
        if project.xdsl_relative:
            xdsl_path = (path.parent / project.xdsl_relative).resolve()
            if not xdsl_path.is_file():
                raise ModelIOError(
                    f"XDSL referenced by project not found: {xdsl_path}"
                )
            self.open_xdsl(xdsl_path)
        return project
