"""Structural DTOs describing a loaded Bayesian network."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeKind(str, Enum):
    DISCRETE = "discrete"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NodeModel:
    id: str
    kind: NodeKind
    states: tuple[str, ...]
    parents: tuple[str, ...]

    @property
    def num_states(self) -> int:
        return len(self.states)


@dataclass(frozen=True, slots=True)
class EdgeModel:
    parent: str
    child: str

    @property
    def id(self) -> str:
        return f"{self.parent}->{self.child}"
