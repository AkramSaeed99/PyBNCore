"""Compile and single-node query operations (Phase 1)."""
from __future__ import annotations

import logging
from typing import Mapping

from pybncore_gui.domain.errors import CompileError, QueryError
from pybncore_gui.domain.results import CompileStats, PosteriorResult
from pybncore_gui.domain.session import ModelSession

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session

    def compile(self) -> CompileStats:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            try:
                wrapper.clear_evidence()
                wrapper.update_beliefs()
            except Exception as e:  # noqa: BLE001
                logger.exception("Compile failed")
                raise CompileError(str(e)) from e
            self._session.mark_compiled()
            node_ids = list(wrapper.nodes())
            edge_count = sum(len(wrapper.parents(n)) for n in node_ids)
            return CompileStats(node_count=len(node_ids), edge_count=edge_count)

    def query_single(
        self,
        node: str,
        evidence: Mapping[str, str],
    ) -> PosteriorResult:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                self._apply_evidence(wrapper, evidence)
                wrapper.update_beliefs()
                self._session.mark_compiled()
                states = tuple(wrapper.get_outcomes(node))
                probs = tuple(float(wrapper.query_p(node, s)) for s in states)
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
            )

    @staticmethod
    def _apply_evidence(wrapper, evidence: Mapping[str, str]) -> None:
        wrapper.clear_evidence()
        if not evidence:
            return
        for key, value in evidence.items():
            if key not in wrapper.nodes():
                raise QueryError(f"Unknown node in evidence: {key}")
            outcomes = list(wrapper.get_outcomes(key))
            if value not in outcomes:
                raise QueryError(
                    f"State '{value}' is not a valid outcome for '{key}'. "
                    f"Valid: {outcomes}"
                )
        wrapper.set_evidence(dict(evidence))
