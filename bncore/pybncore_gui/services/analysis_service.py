"""Sensitivity, VOI, benchmark, and Monte-Carlo analysis."""
from __future__ import annotations

import logging
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from pybncore_gui.domain.errors import QueryError
from pybncore_gui.domain.results import (
    BenchmarkPoint,
    BenchmarkResult,
    MonteCarloResult,
    MonteCarloSummary,
    SensitivityEntry,
    SensitivityReport,
    VOIEntry,
    VOIReport,
)
from pybncore_gui.domain.session import ModelSession

logger = logging.getLogger(__name__)

_ProgressCb = Callable[[int, str], None]


class AnalysisService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session

    # ---------------------------------------------------------- sensitivity

    def sensitivity_ranking(
        self,
        query_node: str,
        query_state: str,
        *,
        n_top: int,
        epsilon: float,
        evidence: Mapping[str, str] | None = None,
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
    ) -> SensitivityReport:
        evidence = evidence or {}
        soft_evidence = soft_evidence or {}
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                self._apply_evidence(wrapper, evidence, soft_evidence)
                raw = wrapper.sensitivity_ranking(
                    query_node, query_state, int(n_top), float(epsilon)
                )
            except QueryError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("Sensitivity ranking failed")
                raise QueryError(str(e)) from e

            entries = tuple(
                SensitivityEntry(
                    target_node=str(t_node),
                    parent_config=tuple(str(p) for p in parent_cfg),
                    target_state=str(t_state),
                    score=float(score),
                )
                for (t_node, parent_cfg, t_state, score) in raw
            )
            return SensitivityReport(
                query_node=query_node,
                query_state=query_state,
                n_top=int(n_top),
                epsilon=float(epsilon),
                entries=entries,
                evidence_snapshot=dict(evidence),
            )

    # ------------------------------------------------------------------ VOI

    def value_of_information(
        self,
        query_node: str,
        candidate_nodes: Sequence[str] | None = None,
        *,
        evidence: Mapping[str, str] | None = None,
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
    ) -> VOIReport:
        evidence = evidence or {}
        soft_evidence = soft_evidence or {}
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                self._apply_evidence(wrapper, evidence, soft_evidence)
                raw = wrapper.value_of_information(
                    query_node,
                    list(candidate_nodes) if candidate_nodes else None,
                )
            except QueryError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("VOI failed")
                raise QueryError(str(e)) from e
            entries = tuple(
                VOIEntry(candidate=str(c), score=float(s)) for (c, s) in raw
            )
            return VOIReport(
                query_node=query_node,
                entries=entries,
                evidence_snapshot=dict(evidence),
            )

    # ---------------------------------------------------------- benchmark

    def benchmark(
        self,
        query_nodes: Sequence[str],
        observed_nodes: Sequence[str],
        row_counts: Sequence[int],
        *,
        seed: int | None = None,
        progress: _ProgressCb | None = None,
    ) -> BenchmarkResult:
        if not query_nodes:
            raise QueryError("Select at least one query node.")
        row_counts = [int(x) for x in row_counts if int(x) > 0]
        if not row_counts:
            raise QueryError("Provide at least one positive row count.")
        rng = np.random.default_rng(seed)

        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                known = set(wrapper.nodes())
                for q in query_nodes:
                    if q not in known:
                        raise QueryError(f"Unknown query node: {q}")
                for o in observed_nodes:
                    if o not in known:
                        raise QueryError(f"Unknown observed node: {o}")

                # Warm-up compile so the first timed run is fair.
                wrapper.clear_evidence()
                wrapper.clear_soft_evidence()
                wrapper.update_beliefs()

                points: list[BenchmarkPoint] = []
                for idx, n_rows in enumerate(row_counts):
                    rows = self._random_rows(wrapper, observed_nodes, n_rows, rng)
                    matrix = (
                        wrapper.make_evidence_matrix(rows, list(observed_nodes))
                        if observed_nodes
                        else None
                    )
                    if progress is not None:
                        progress(
                            int(10 + 80 * idx / max(1, len(row_counts))),
                            f"Timing {n_rows} rows…",
                        )
                    t0 = time.perf_counter()
                    wrapper.batch_query_marginals(list(query_nodes), matrix)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    points.append(
                        BenchmarkPoint(
                            num_rows=n_rows,
                            elapsed_ms=float(elapsed),
                            ms_per_row=float(elapsed) / max(1, n_rows),
                        )
                    )
            except QueryError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("Benchmark failed")
                raise QueryError(str(e)) from e

        return BenchmarkResult(
            query_nodes=tuple(query_nodes),
            observed_nodes=tuple(observed_nodes),
            points=tuple(points),
        )

    # --------------------------------------------------------- Monte Carlo

    def monte_carlo(
        self,
        query_nodes: Sequence[str],
        observed_nodes: Sequence[str],
        *,
        num_samples: int,
        seed: int | None = None,
        progress: _ProgressCb | None = None,
    ) -> MonteCarloResult:
        if num_samples <= 0:
            raise QueryError("Number of samples must be positive.")
        if not query_nodes:
            raise QueryError("Select at least one query node.")
        rng = np.random.default_rng(seed)

        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                known = set(wrapper.nodes())
                for q in query_nodes:
                    if q not in known:
                        raise QueryError(f"Unknown query node: {q}")
                for o in observed_nodes:
                    if o not in known:
                        raise QueryError(f"Unknown observed node: {o}")

                wrapper.clear_evidence()
                wrapper.clear_soft_evidence()

                rows = self._random_rows(wrapper, observed_nodes, num_samples, rng)
                if progress is not None:
                    progress(20, f"Sampling {num_samples} evidence rows…")

                matrix = (
                    wrapper.make_evidence_matrix(rows, list(observed_nodes))
                    if observed_nodes
                    else None
                )
                if progress is not None:
                    progress(40, "Running batch query…")
                marginals_raw = wrapper.batch_query_marginals(list(query_nodes), matrix)

                summaries: dict[str, MonteCarloSummary] = {}
                for node in query_nodes:
                    states = tuple(wrapper.get_outcomes(node))
                    raw = marginals_raw[node]
                    if isinstance(raw, dict):
                        arr = np.stack(
                            [np.asarray(raw[s], dtype=np.float64) for s in states],
                            axis=-1,
                        )
                    else:
                        arr = np.asarray(raw, dtype=np.float64)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, len(states))
                    mean = arr.mean(axis=0)
                    std = arr.std(axis=0)
                    summaries[node] = MonteCarloSummary(
                        node=node,
                        states=states,
                        mean=tuple(float(x) for x in mean.tolist()),
                        std=tuple(float(x) for x in std.tolist()),
                    )
                if progress is not None:
                    progress(100, "Monte Carlo complete")
            except QueryError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("Monte Carlo failed")
                raise QueryError(str(e)) from e

        return MonteCarloResult(
            query_nodes=tuple(query_nodes),
            observed_nodes=tuple(observed_nodes),
            num_samples=int(num_samples),
            summaries=summaries,
        )

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _random_rows(
        wrapper,
        observed_nodes: Sequence[str],
        n_rows: int,
        rng: np.random.Generator,
    ) -> list[dict[str, str]]:
        if not observed_nodes:
            return [{} for _ in range(n_rows)]
        options: dict[str, list[str]] = {
            n: list(wrapper.get_outcomes(n)) for n in observed_nodes
        }
        rows: list[dict[str, str]] = []
        for _ in range(n_rows):
            row = {n: str(rng.choice(opts)) for n, opts in options.items()}
            rows.append(row)
        return rows

    @staticmethod
    def _apply_evidence(
        wrapper,
        evidence: Mapping[str, str],
        soft_evidence: Mapping[str, Mapping[str, float]],
    ) -> None:
        wrapper.clear_evidence()
        wrapper.clear_soft_evidence()
        known = set(wrapper.nodes())
        if evidence:
            for k, v in evidence.items():
                if k not in known:
                    raise QueryError(f"Unknown node in evidence: {k}")
                outcomes = list(wrapper.get_outcomes(k))
                if v not in outcomes:
                    raise QueryError(f"State '{v}' not valid for '{k}'.")
            wrapper.set_evidence(dict(evidence))
        for node, likelihoods in soft_evidence.items():
            if node not in known:
                raise QueryError(f"Unknown node in soft evidence: {node}")
            outcomes = list(wrapper.get_outcomes(node))
            vec: dict[str, float] = {}
            for state, value in likelihoods.items():
                if state not in outcomes:
                    raise QueryError(
                        f"State '{state}' not valid for '{node}' in soft evidence."
                    )
                v = float(value)
                if not np.isfinite(v) or v < 0:
                    raise QueryError(f"Soft evidence for '{node}' must be ≥ 0.")
                vec[state] = v
            if vec and sum(vec.values()) <= 0:
                raise QueryError(f"Soft evidence for '{node}' must have positive mass.")
            if vec:
                wrapper.set_soft_evidence(node, vec)
