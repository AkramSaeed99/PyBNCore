"""Project sidecar — persistent GUI metadata beside an XDSL."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

PROJECT_VERSION = 1


@dataclass(slots=True)
class ProjectFile:
    xdsl_relative: str | None = None
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    scenarios: list[dict] = field(default_factory=list)   # Phase 3 placeholder
    version: int = PROJECT_VERSION

    def to_json(self) -> str:
        payload = asdict(self)
        payload["positions"] = {k: list(v) for k, v in self.positions.items()}
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProjectFile":
        data = json.loads(text)
        positions = {
            k: (float(v[0]), float(v[1])) for k, v in data.get("positions", {}).items()
        }
        return cls(
            xdsl_relative=data.get("xdsl_relative"),
            positions=positions,
            scenarios=list(data.get("scenarios", [])),
            version=int(data.get("version", PROJECT_VERSION)),
        )

    @classmethod
    def read(cls, path: str | Path) -> "ProjectFile":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def update_positions(self, positions: Mapping[str, tuple[float, float]]) -> None:
        self.positions = {k: (float(v[0]), float(v[1])) for k, v in positions.items()}
