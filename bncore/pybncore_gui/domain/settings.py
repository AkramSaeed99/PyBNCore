"""Engine settings — controls triangulation heuristic and loopy-BP mode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

TRIANGULATION_HEURISTICS: tuple[str, ...] = ("min_fill", "min_degree")


@dataclass(slots=True)
class EngineSettings:
    use_loopy_bp: bool = False
    loopy_iterations: int = 100
    loopy_damping: float = 0.5
    loopy_tolerance: float = 1e-6
    triangulation: str = "min_fill"

    def validated(self) -> "EngineSettings":
        if self.triangulation not in TRIANGULATION_HEURISTICS:
            self.triangulation = "min_fill"
        self.loopy_iterations = max(1, int(self.loopy_iterations))
        self.loopy_damping = float(min(0.99, max(0.0, self.loopy_damping)))
        self.loopy_tolerance = float(max(1e-12, self.loopy_tolerance))
        return self

    def to_dict(self) -> dict:
        return {
            "use_loopy_bp": bool(self.use_loopy_bp),
            "loopy_iterations": int(self.loopy_iterations),
            "loopy_damping": float(self.loopy_damping),
            "loopy_tolerance": float(self.loopy_tolerance),
            "triangulation": str(self.triangulation),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "EngineSettings":
        return cls(
            use_loopy_bp=bool(data.get("use_loopy_bp", False)),
            loopy_iterations=int(data.get("loopy_iterations", 100)),
            loopy_damping=float(data.get("loopy_damping", 0.5)),
            loopy_tolerance=float(data.get("loopy_tolerance", 1e-6)),
            triangulation=str(data.get("triangulation", "min_fill")),
        ).validated()
