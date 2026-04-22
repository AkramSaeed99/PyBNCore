"""Specs for continuous / deterministic nodes and rare-event thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


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
    # Python source for `fn` — stored so deterministic nodes can round-trip
    # through the project sidecar (callables themselves cannot).
    fn_source: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "parents": list(self.parents),
            "domain": list(self.domain),
            "initial_bins": int(self.initial_bins),
            "rare_event_mode": bool(self.rare_event_mode),
            "log_spaced": bool(self.log_spaced),
            "monotone": bool(self.monotone),
            "n_samples": int(self.n_samples),
            "params": dict(self.params),
            "fn_source": self.fn_source,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ContinuousNodeSpec":
        return cls(
            name=str(data["name"]),
            kind=ContinuousDistKind(str(data["kind"])),
            parents=tuple(str(p) for p in data.get("parents", ())),
            domain=tuple(float(x) for x in data.get("domain", (0.0, 1.0))),  # type: ignore[arg-type]
            initial_bins=int(data.get("initial_bins", 8)),
            rare_event_mode=bool(data.get("rare_event_mode", False)),
            log_spaced=bool(data.get("log_spaced", False)),
            monotone=bool(data.get("monotone", False)),
            n_samples=int(data.get("n_samples", 32)),
            params=dict(data.get("params", {})),
            fn=None,
            fn_source=(
                str(data["fn_source"]) if data.get("fn_source") is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    node: str
    threshold: float
