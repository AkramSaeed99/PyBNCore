"""Named evidence bundles (hard + soft) saved in the project sidecar."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(slots=True)
class Scenario:
    name: str
    evidence: dict[str, str] = field(default_factory=dict)
    soft_evidence: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "evidence": dict(self.evidence),
            "soft_evidence": {
                k: dict(v) for k, v in self.soft_evidence.items()
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Scenario":
        return cls(
            name=str(data.get("name", "")),
            evidence={
                str(k): str(v) for k, v in (data.get("evidence") or {}).items()
            },
            soft_evidence={
                str(n): {str(s): float(p) for s, p in (d or {}).items()}
                for n, d in (data.get("soft_evidence") or {}).items()
            },
            notes=str(data.get("notes", "")),
        )
