"""Project sidecar — persistent GUI metadata beside an XDSL."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from pybncore_gui.domain.continuous import ContinuousNodeSpec
from pybncore_gui.domain.scenario import Scenario
from pybncore_gui.domain.settings import EngineSettings
from pybncore_gui.domain.submodel import SubModelLayout

PROJECT_VERSION = 5


@dataclass(slots=True)
class ProjectFile:
    xdsl_relative: str | None = None
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    scenarios: list[Scenario] = field(default_factory=list)
    settings: EngineSettings = field(default_factory=EngineSettings)
    layout: SubModelLayout = field(default_factory=SubModelLayout)
    descriptions: dict[str, str] = field(default_factory=dict)
    continuous_specs: list[ContinuousNodeSpec] = field(default_factory=list)
    equation_sources: dict[str, dict] = field(default_factory=dict)
    version: int = PROJECT_VERSION

    def to_json(self) -> str:
        payload = {
            "version": self.version,
            "xdsl_relative": self.xdsl_relative,
            "positions": {k: list(v) for k, v in self.positions.items()},
            "scenarios": [s.to_dict() for s in self.scenarios],
            "settings": self.settings.to_dict(),
            "layout": self.layout.to_dict(),
            "descriptions": dict(self.descriptions),
            "continuous_specs": [s.to_dict() for s in self.continuous_specs],
            "equation_sources": {k: dict(v) for k, v in self.equation_sources.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProjectFile":
        data = json.loads(text)
        positions = {
            k: (float(v[0]), float(v[1])) for k, v in data.get("positions", {}).items()
        }
        scenarios_raw = data.get("scenarios", []) or []
        scenarios: list[Scenario] = []
        for s in scenarios_raw:
            if isinstance(s, dict):
                scenarios.append(Scenario.from_dict(s))
        settings_raw = data.get("settings")
        settings = (
            EngineSettings.from_dict(settings_raw)
            if isinstance(settings_raw, Mapping)
            else EngineSettings()
        )
        layout_raw = data.get("layout")
        layout = (
            SubModelLayout.from_dict(layout_raw)
            if isinstance(layout_raw, Mapping)
            else SubModelLayout()
        )
        descriptions = {
            str(k): str(v) for k, v in (data.get("descriptions") or {}).items()
        }
        continuous_specs_raw = data.get("continuous_specs") or []
        continuous_specs = [
            ContinuousNodeSpec.from_dict(s)
            for s in continuous_specs_raw
            if isinstance(s, Mapping)
        ]
        equation_sources = {
            str(k): dict(v)
            for k, v in (data.get("equation_sources") or {}).items()
            if isinstance(v, Mapping)
        }
        return cls(
            xdsl_relative=data.get("xdsl_relative"),
            positions=positions,
            scenarios=scenarios,
            settings=settings,
            layout=layout,
            descriptions=descriptions,
            continuous_specs=continuous_specs,
            equation_sources=equation_sources,
            version=int(data.get("version", PROJECT_VERSION)),
        )

    @classmethod
    def read(cls, path: str | Path) -> "ProjectFile":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def update_positions(self, positions: Mapping[str, tuple[float, float]]) -> None:
        self.positions = {k: (float(v[0]), float(v[1])) for k, v in positions.items()}
