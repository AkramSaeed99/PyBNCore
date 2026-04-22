"""Specs for continuous / deterministic nodes and rare-event thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ContinuousDistKind(str, Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    DETERMINISTIC = "deterministic"


@dataclass(slots=True)
class ContinuousNodeSpec:
    name: str
    kind: ContinuousDistKind
    parents: tuple[str, ...] = ()
    domain: tuple[float, float] = (0.0, 1.0)
    initial_bins: int = 8
    rare_event_mode: bool = False
    log_spaced: bool = False
    monotone: bool = False
    n_samples: int = 32
    params: dict[str, Any] = field(default_factory=dict)
    # For DETERMINISTIC nodes the callable lives here; for the others it is
    # absent and `params` holds plain scalars.
    fn: Callable[..., float] | None = None


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    node: str
    threshold: float
