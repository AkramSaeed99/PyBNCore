"""Structural validation — cycles, missing CPTs, shape issues, isolated nodes."""
from __future__ import annotations

import numpy as np

from pybncore_gui.domain.session import ModelSession
from pybncore_gui.domain.validation import Severity, ValidationIssue, ValidationReport


class ValidationService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session

    def validate(self) -> ValidationReport:
        issues: list[ValidationIssue] = []
        with self._session.locked() as wrapper:
            if wrapper is None:
                return ValidationReport()
            nodes = list(wrapper.nodes())
            if not nodes:
                issues.append(
                    ValidationIssue(
                        Severity.INFO, "empty", "Model has no nodes."
                    )
                )
                return ValidationReport(tuple(issues))

            parents_of = {n: list(wrapper.parents(n)) for n in nodes}
            children_of = {n: list(wrapper.children(n)) for n in nodes}

            cycle = self._detect_cycle(nodes, children_of)
            if cycle:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        "cycle",
                        f"Directed cycle detected: {' → '.join(cycle)} → {cycle[0]}",
                    )
                )

            for n in nodes:
                states = list(wrapper.get_outcomes(n))
                if len(states) < 2:
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            "too_few_states",
                            f"Node '{n}' has fewer than 2 states.",
                            node=n,
                        )
                    )
                raw = wrapper._cpts.get(n)
                if raw is None or np.asarray(raw).size == 0:
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            "missing_cpt",
                            f"Node '{n}' has no CPT.",
                            node=n,
                        )
                    )
                    continue
                flat = np.asarray(raw, dtype=np.float64)
                n_states = len(states)
                if n_states == 0 or flat.size % n_states != 0:
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            "cpt_shape",
                            f"CPT for '{n}' has size {flat.size}, not a multiple of {n_states} states.",
                            node=n,
                        )
                    )
                    continue
                shaped = flat.reshape(-1, n_states)
                row_sums = shaped.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-6):
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            "cpt_rows",
                            f"CPT rows for '{n}' do not all sum to 1.0.",
                            node=n,
                        )
                    )
                if np.any(shaped < -1e-12) or np.any(shaped > 1 + 1e-12):
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            "cpt_range",
                            f"CPT for '{n}' contains values outside [0, 1].",
                            node=n,
                        )
                    )

                if not parents_of[n] and not children_of[n]:
                    issues.append(
                        ValidationIssue(
                            Severity.INFO,
                            "isolated",
                            f"Node '{n}' is not connected to any other node.",
                            node=n,
                        )
                    )

        return ValidationReport(tuple(issues))

    @staticmethod
    def _detect_cycle(
        nodes: list[str], children_of: dict[str, list[str]]
    ) -> list[str] | None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in nodes}
        parent: dict[str, str | None] = {n: None for n in nodes}

        def dfs(start: str) -> list[str] | None:
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                node, child_idx = stack[-1]
                if color[node] == WHITE:
                    color[node] = GRAY
                kids = children_of.get(node, [])
                if child_idx < len(kids):
                    stack[-1] = (node, child_idx + 1)
                    nxt = kids[child_idx]
                    if color[nxt] == GRAY:
                        # reconstruct cycle path: walk parent chain from `node` back to `nxt`.
                        path = [nxt]
                        cur = node
                        while cur is not None and cur != nxt:
                            path.append(cur)
                            cur = parent.get(cur)
                        return list(reversed(path))
                    if color[nxt] == WHITE:
                        parent[nxt] = node
                        stack.append((nxt, 0))
                else:
                    color[node] = BLACK
                    stack.pop()
            return None

        for n in nodes:
            if color[n] == WHITE:
                cycle = dfs(n)
                if cycle:
                    return cycle
        return None
