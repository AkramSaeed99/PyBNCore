"""Compile, single-node query, MAP, batch, JT stats, engine settings."""
from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
from pybncore import BatchExecutionEngine, JunctionTreeCompiler
from pybncore.loopy_bp import LoopyBPEngine

from pybncore_gui.domain.errors import CompileError, QueryError
from pybncore_gui.domain.results import (
    BatchQueryResult,
    CompileStats,
    ContinuousPosteriorDTO,
    HybridResultDTO,
    JTStats,
    MAPResult,
    PosteriorResult,
)
from pybncore_gui.domain.session import ModelSession
from pybncore_gui.domain.settings import EngineSettings

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session
        self._settings = EngineSettings()

    # ----------------------------------------------------------- settings

    @property
    def settings(self) -> EngineSettings:
        return self._settings

    def update_settings(self, settings: EngineSettings) -> None:
        self._settings = settings.validated()
        self._session.invalidate_compile()

    # --------------------------------------------------------------- compile

    def compile(self) -> CompileStats:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            try:
                self._ensure_jt(wrapper)
            except CompileError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("Compile failed")
                raise CompileError(str(e)) from e
            self._session.mark_compiled()
            node_ids = list(wrapper.nodes())
            edge_count = sum(len(wrapper.parents(n)) for n in node_ids)
            return CompileStats(node_count=len(node_ids), edge_count=edge_count)

    def compute_jt_stats(self) -> JTStats:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            try:
                self._ensure_jt(wrapper)
                stats = wrapper._jt.stats()
            except Exception as e:  # noqa: BLE001
                raise CompileError(f"JT stats unavailable: {e}") from e
            return JTStats(
                treewidth=int(stats.treewidth),
                num_cliques=int(stats.num_cliques),
                max_clique_size=int(stats.max_clique_size),
                total_table_entries=int(stats.total_table_entries),
            )

    # ----------------------------------------------------- single-node query

    def query_single(
        self,
        node: str,
        evidence: Mapping[str, str],
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
    ) -> PosteriorResult:
        soft_evidence = soft_evidence or {}
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                states = tuple(wrapper.get_outcomes(node))
                if self._settings.use_loopy_bp:
                    probs = self._loopy_marginals_locked(wrapper, node, evidence, soft_evidence, states)
                    label = "loopy_bp"
                else:
                    self._apply_evidence(wrapper, evidence, soft_evidence)
                    self._ensure_jt(wrapper)
                    wrapper.update_beliefs()
                    probs = tuple(float(wrapper.query_p(node, s)) for s in states)
                    label = "exact"
                self._session.mark_compiled()
            except QueryError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("Query failed: node=%s", node)
                raise QueryError(str(e)) from e

            return PosteriorResult(
                node=node,
                states=states,
                probabilities=probs,
                evidence_snapshot=dict(evidence),
                soft_evidence_snapshot={k: dict(v) for k, v in soft_evidence.items()},
                engine_label=label,
            )

    # ------------------------------------------------------------------- MAP

    def query_map(
        self,
        evidence: Mapping[str, str],
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
    ) -> MAPResult:
        soft_evidence = soft_evidence or {}
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                if self._settings.use_loopy_bp:
                    assignment = self._loopy_map_locked(wrapper, evidence, soft_evidence)
                else:
                    self._apply_evidence(wrapper, evidence, soft_evidence)
                    self._ensure_jt(wrapper)
                    raw = wrapper.query_map()
                    assignment = {str(k): str(v) for k, v in raw.items()}
            except QueryError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("MAP query failed")
                raise QueryError(str(e)) from e

            return MAPResult(
                assignment=assignment,
                evidence_snapshot=dict(evidence),
                soft_evidence_snapshot={k: dict(v) for k, v in soft_evidence.items()},
            )

    # ----------------------------------------------------------------- batch

    def batch_query(
        self,
        query_nodes: Sequence[str],
        evidence_rows: Sequence[Mapping[str, str]],
    ) -> BatchQueryResult:
        if self._settings.use_loopy_bp:
            raise QueryError(
                "Batch queries are not supported in loopy-BP mode. "
                "Disable loopy BP in Engine Settings to run batch."
            )
        if not query_nodes:
            raise QueryError("Select at least one node to query.")
        if not evidence_rows:
            raise QueryError("Batch evidence table is empty.")

        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                evidence_columns = sorted({
                    k for row in evidence_rows for k in row.keys()
                })
                self._validate_evidence_columns(wrapper, evidence_columns)
                matrix = self._build_evidence_matrix(wrapper, evidence_rows)
                self._ensure_jt(wrapper)
                marginals_raw = wrapper.batch_query_marginals(list(query_nodes), matrix)
            except Exception as e:  # noqa: BLE001
                logger.exception("Batch query failed")
                raise QueryError(str(e)) from e

            state_labels: dict[str, tuple[str, ...]] = {}
            marginals: dict[str, np.ndarray] = {}
            for node in query_nodes:
                states = tuple(wrapper.get_outcomes(node))
                state_labels[node] = states
                value = marginals_raw[node]
                if isinstance(value, dict):
                    # Convert dict-of-states → ndarray[n_rows, n_states]
                    arr = np.stack(
                        [np.asarray(value[s], dtype=np.float64) for s in states], axis=-1
                    )
                else:
                    arr = np.asarray(value, dtype=np.float64)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, len(states))
                marginals[node] = arr

            return BatchQueryResult(
                nodes=tuple(query_nodes),
                evidence_columns=tuple(evidence_columns),
                marginals=marginals,
                state_labels=state_labels,
                num_rows=len(evidence_rows),
            )

    # ----------------------------------------------------- evidence helpers

    def _apply_evidence(
        self,
        wrapper,
        evidence: Mapping[str, str],
        soft_evidence: Mapping[str, Mapping[str, float]],
    ) -> None:
        wrapper.clear_evidence()
        wrapper.clear_soft_evidence()
        known = set(wrapper.nodes())
        for key, value in evidence.items():
            if key not in known:
                raise QueryError(f"Unknown node in evidence: {key}")
            outcomes = list(wrapper.get_outcomes(key))
            if value not in outcomes:
                raise QueryError(
                    f"State '{value}' is not a valid outcome for '{key}'. Valid: {outcomes}"
                )
        if evidence:
            wrapper.set_evidence(dict(evidence))
        for node, likelihoods in soft_evidence.items():
            if node not in known:
                raise QueryError(f"Unknown node in soft evidence: {node}")
            outcomes = list(wrapper.get_outcomes(node))
            vec = {}
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

    # ------------------------------------------------------------ hybrid

    def run_hybrid(
        self,
        query_nodes: Sequence[str],
        *,
        max_iters: int = 8,
        eps_entropy: float = 1e-4,
        eps_kl: float = 1e-4,
        evidence: Mapping[str, str] | None = None,
        soft_evidence: Mapping[str, Mapping[str, float]] | None = None,
        continuous_evidence: Mapping[str, float] | None = None,
        continuous_likelihoods: Mapping[str, object] | None = None,
        pdf_samples: int = 256,
        quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> HybridResultDTO:
        evidence = dict(evidence or {})
        soft_evidence = dict(soft_evidence or {})
        continuous_evidence = dict(continuous_evidence or {})
        continuous_likelihoods = dict(continuous_likelihoods or {})

        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            if not query_nodes:
                raise QueryError("Select at least one node to query.")
            try:
                self._apply_evidence(wrapper, evidence, soft_evidence)
                wrapper.clear_continuous_evidence()
                if continuous_evidence:
                    wrapper.set_continuous_evidence(dict(continuous_evidence))
                for name, fn in continuous_likelihoods.items():
                    if fn is None:
                        continue
                    wrapper.set_continuous_likelihood(name, fn)

                raw = wrapper.hybrid_query(
                    list(query_nodes),
                    max_iters=int(max_iters),
                    eps_entropy=float(eps_entropy),
                    eps_kl=float(eps_kl),
                )
            except QueryError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.exception("Hybrid query failed")
                raise QueryError(str(e)) from e

            continuous_posteriors: dict[str, ContinuousPosteriorDTO] = {}
            discrete_posteriors: dict[str, PosteriorResult] = {}

            posteriors = getattr(raw, "_posteriors", None) or {}
            for node in query_nodes:
                value = posteriors.get(node)
                if value is None:
                    continue
                # Discrete nodes in hybrid queries come back as {state: prob}.
                if isinstance(value, Mapping):
                    states = tuple(str(k) for k in value.keys())
                    probs = tuple(float(value[s]) for s in states)
                    discrete_posteriors[node] = PosteriorResult(
                        node=node,
                        states=states,
                        probabilities=probs,
                        evidence_snapshot=dict(evidence),
                        soft_evidence_snapshot={k: dict(v) for k, v in soft_evidence.items()},
                        engine_label="hybrid",
                    )
                else:
                    continuous_posteriors[node] = self._build_continuous_dto(
                        value, pdf_samples, quantiles
                    )

            return HybridResultDTO(
                query_nodes=tuple(query_nodes),
                iterations_used=int(getattr(raw, "iterations_used", 0)),
                max_iters=int(getattr(raw, "max_iters", max_iters)),
                final_max_error=float(getattr(raw, "final_max_error", 0.0)),
                converged=bool(getattr(raw, "converged", False)),
                continuous=continuous_posteriors,
                discrete=discrete_posteriors,
            )

    @staticmethod
    def _build_continuous_dto(
        posterior,
        pdf_samples: int,
        quantiles: Sequence[float],
    ) -> ContinuousPosteriorDTO:
        def _scalar(value) -> float:
            return float(value() if callable(value) else value)

        def _seq(value) -> tuple[float, ...]:
            raw = value() if callable(value) else value
            return tuple(float(x) for x in raw)

        support_raw = posterior.support
        support = tuple(
            float(x) for x in (support_raw() if callable(support_raw) else support_raw)
        )
        lo, hi = support
        edges = _seq(getattr(posterior, "edges"))
        masses = _seq(getattr(posterior, "bin_masses"))

        grid = np.linspace(lo, hi, max(32, int(pdf_samples)))
        pdf_grid = tuple((float(x), float(posterior.pdf(float(x)))) for x in grid)
        cdf_grid = tuple((float(x), float(posterior.cdf(float(x)))) for x in grid)
        try:
            median = _scalar(posterior.median)
        except Exception:  # noqa: BLE001
            median = float(posterior.quantile(0.5))
        quantile_pairs = tuple(
            (float(q), float(posterior.quantile(float(q)))) for q in quantiles
        )
        num_bins_raw = posterior.num_bins
        return ContinuousPosteriorDTO(
            name=str(getattr(posterior, "name", "")),
            num_bins=int(num_bins_raw() if callable(num_bins_raw) else num_bins_raw),
            support=support,
            bin_edges=edges,
            bin_masses=masses,
            mean=_scalar(posterior.mean),
            std=_scalar(posterior.std),
            median=median,
            pdf_grid=pdf_grid,
            cdf_grid=cdf_grid,
            quantiles=quantile_pairs,
        )

    @staticmethod
    def _build_evidence_matrix(wrapper, rows: Sequence[Mapping[str, str]]) -> np.ndarray:
        num_vars = wrapper._graph.num_variables()
        name_to_id = wrapper._name_to_id
        node_states = wrapper._node_states
        matrix = np.full((len(rows), num_vars), -1, dtype=np.int32)
        for r, row in enumerate(rows):
            for name, value in row.items():
                vid = name_to_id.get(name)
                if vid is None:
                    continue
                if isinstance(value, int):
                    idx = int(value)
                else:
                    states = node_states.get(name, [])
                    try:
                        idx = states.index(str(value))
                    except ValueError:
                        continue
                matrix[r, vid] = idx
        return matrix

    @staticmethod
    def _validate_evidence_columns(wrapper, columns: Sequence[str]) -> None:
        known = set(wrapper.nodes())
        for col in columns:
            if col not in known:
                raise QueryError(f"Unknown node in batch evidence: '{col}'")

    # ---------------------------------------------------- junction tree mgmt

    def _ensure_jt(self, wrapper) -> None:
        if self._session.is_compiled and wrapper._jt is not None:
            return
        heuristic = self._settings.triangulation or "min_fill"
        jt = JunctionTreeCompiler.compile(wrapper._graph, heuristic)
        wrapper._jt = jt
        wrapper._engine = BatchExecutionEngine(jt, 0, 1024)
        wrapper._is_compiled = True

    # ---------------------------------------------------------- loopy BP

    def _build_loopy(self, wrapper) -> LoopyBPEngine:
        cpts_shaped: dict[str, np.ndarray] = {}
        for name in wrapper.nodes():
            flat = wrapper._cpts.get(name)
            if flat is None:
                raise QueryError(f"Missing CPT for '{name}'")
            cpts_shaped[name] = np.asarray(flat, dtype=np.float64)
        return LoopyBPEngine(
            wrapper._graph,
            cpts_shaped,
            damping=self._settings.loopy_damping,
            max_iterations=self._settings.loopy_iterations,
            tolerance=self._settings.loopy_tolerance,
        )

    def _loopy_marginals_locked(
        self,
        wrapper,
        node: str,
        evidence: Mapping[str, str],
        soft_evidence: Mapping[str, Mapping[str, float]],
        states: Sequence[str],
    ) -> tuple[float, ...]:
        if soft_evidence:
            raise QueryError(
                "Soft evidence is not supported by the loopy-BP fallback in Phase 3."
            )
        engine = self._build_loopy(wrapper)
        result = engine.infer(evidence=dict(evidence) if evidence else None)
        arr = np.asarray(result[node], dtype=np.float64)
        if arr.size != len(states):
            raise QueryError(
                f"Loopy BP returned wrong number of states for '{node}'."
            )
        return tuple(float(x) for x in arr.tolist())

    def _loopy_map_locked(
        self,
        wrapper,
        evidence: Mapping[str, str],
        soft_evidence: Mapping[str, Mapping[str, float]],
    ) -> dict[str, str]:
        if soft_evidence:
            raise QueryError("Soft evidence is not supported by loopy-BP.")
        engine = self._build_loopy(wrapper)
        result = engine.infer(evidence=dict(evidence) if evidence else None)
        assignment: dict[str, str] = {}
        for node in wrapper.nodes():
            states = list(wrapper.get_outcomes(node))
            arr = np.asarray(result[node], dtype=np.float64)
            idx = int(np.argmax(arr))
            assignment[node] = states[idx]
        # Observed nodes take their observed state.
        for k, v in evidence.items():
            if k in assignment:
                assignment[k] = v
        return assignment
