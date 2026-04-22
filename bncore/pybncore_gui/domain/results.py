"""Typed result objects returned from service calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class CompileStats:
    node_count: int
    edge_count: int


@dataclass(frozen=True, slots=True)
class JTStats:
    treewidth: int
    num_cliques: int
    max_clique_size: int
    total_table_entries: int


@dataclass(frozen=True, slots=True)
class PosteriorResult:
    node: str
    states: tuple[str, ...]
    probabilities: tuple[float, ...]
    evidence_snapshot: Mapping[str, str] = field(default_factory=dict)
    soft_evidence_snapshot: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    engine_label: str = "exact"

    def __post_init__(self) -> None:
        if len(self.states) != len(self.probabilities):
            raise ValueError(
                f"states ({len(self.states)}) and probabilities "
                f"({len(self.probabilities)}) length mismatch"
            )


@dataclass(frozen=True, slots=True)
class MAPResult:
    assignment: Mapping[str, str]
    evidence_snapshot: Mapping[str, str] = field(default_factory=dict)
    soft_evidence_snapshot: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BatchQueryResult:
    nodes: tuple[str, ...]
    evidence_columns: tuple[str, ...]
    marginals: Mapping[str, np.ndarray]
    state_labels: Mapping[str, tuple[str, ...]]
    num_rows: int


# --------------------------------------------------------------- sensitivity


@dataclass(frozen=True, slots=True)
class SensitivityEntry:
    target_node: str
    parent_config: tuple[str, ...]
    target_state: str
    score: float


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    query_node: str
    query_state: str
    n_top: int
    epsilon: float
    entries: tuple[SensitivityEntry, ...]
    evidence_snapshot: Mapping[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------- VOI


@dataclass(frozen=True, slots=True)
class VOIEntry:
    candidate: str
    score: float


@dataclass(frozen=True, slots=True)
class VOIReport:
    query_node: str
    entries: tuple[VOIEntry, ...]
    evidence_snapshot: Mapping[str, str] = field(default_factory=dict)


# -------------------------------------------------------------- benchmarks


@dataclass(frozen=True, slots=True)
class BenchmarkPoint:
    num_rows: int
    elapsed_ms: float         # mean across repeats
    ms_per_row: float
    std_ms: float = 0.0       # standard deviation across repeats
    num_repeats: int = 1


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    query_nodes: tuple[str, ...]
    observed_nodes: tuple[str, ...]
    points: tuple[BenchmarkPoint, ...]


# --------------------------------------------------------------- Monte Carlo


@dataclass(frozen=True, slots=True)
class MonteCarloSummary:
    node: str
    states: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    query_nodes: tuple[str, ...]
    observed_nodes: tuple[str, ...]
    num_samples: int
    summaries: Mapping[str, MonteCarloSummary]


# ------------------------------------------------------------- continuous


@dataclass(frozen=True, slots=True)
class ContinuousPosteriorDTO:
    name: str
    num_bins: int
    support: tuple[float, float]
    bin_edges: tuple[float, ...]
    bin_masses: tuple[float, ...]
    mean: float
    std: float
    median: float
    pdf_grid: tuple[tuple[float, float], ...]   # (x, f(x))
    cdf_grid: tuple[tuple[float, float], ...]   # (x, F(x))
    quantiles: tuple[tuple[float, float], ...]  # (q, quantile(q))


@dataclass(frozen=True, slots=True)
class HybridResultDTO:
    query_nodes: tuple[str, ...]
    iterations_used: int
    max_iters: int
    final_max_error: float
    converged: bool
    continuous: Mapping[str, ContinuousPosteriorDTO]
    discrete: Mapping[str, PosteriorResult]
