"""Typed result objects returned from service calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CompileStats:
    node_count: int
    edge_count: int


@dataclass(frozen=True, slots=True)
class PosteriorResult:
    node: str
    states: tuple[str, ...]
    probabilities: tuple[float, ...]
    evidence_snapshot: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.states) != len(self.probabilities):
            raise ValueError(
                f"states ({len(self.states)}) and probabilities "
                f"({len(self.probabilities)}) length mismatch"
            )
