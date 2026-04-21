"""Structural mutations — add/remove/rename nodes and edges, edit CPTs.

The underlying `PyBNCoreWrapper` exposes add-only operations for discrete
structure, so destructive operations (remove, rename, remove-edge) go through
`_rebuild(...)` which snapshots the current graph, mutates the snapshot, and
re-creates the graph in place. CPT shapes are re-initialized when a node's
parent set changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from pybncore import Graph

from pybncore_gui.domain.errors import CompileError, DomainError, EvidenceError
from pybncore_gui.domain.session import ModelSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    name: str
    states: tuple[str, ...]
    parents: tuple[str, ...]
    children: tuple[str, ...]
    cpt_flat: np.ndarray          # raw wrapper._cpts value
    cpt_shaped: np.ndarray        # reshaped (rows, n_states)


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """Full authoring state for undo rebuilds."""

    order: tuple[str, ...]
    states: Mapping[str, tuple[str, ...]]
    parents: Mapping[str, tuple[str, ...]]
    cpts: Mapping[str, np.ndarray]   # flat arrays


class _AuthoringError(DomainError):
    """Raised for bad authoring input (bubbled as EvidenceError or CompileError)."""


class AuthoringService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session

    # ------------------------------------------------------------------ reads

    def snapshot(self) -> GraphSnapshot:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            return self._snapshot_locked(wrapper)

    def node_snapshot(self, name: str) -> NodeSnapshot:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            return self._node_snapshot_locked(wrapper, name)

    # ------------------------------------------------------------------ nodes

    def add_discrete_node(
        self,
        name: str,
        states: Sequence[str],
        parents: Sequence[str] = (),
    ) -> None:
        self._validate_name(name)
        self._validate_states(states)
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            if name in set(wrapper.nodes()):
                raise EvidenceError(f"A node named '{name}' already exists.")
            graph = self._ensure_graph(wrapper)
            for p in parents:
                if p not in set(wrapper.nodes()):
                    raise EvidenceError(f"Parent '{p}' does not exist.")
            graph.add_variable(name, list(states))
            for p in parents:
                graph.add_edge(p, name)
            self._initialise_cpt(wrapper, name, list(states), list(parents))
            wrapper._cache_metadata()
            self._invalidate(wrapper)

    def remove_node(self, name: str) -> NodeSnapshot:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            if name not in set(wrapper.nodes()):
                raise EvidenceError(f"Unknown node: '{name}'")
            snap = self._node_snapshot_locked(wrapper, name)
            full = self._snapshot_locked(wrapper)
            self._rebuild_locked(
                wrapper,
                full,
                exclude_node=name,
            )
            return snap

    def rename_node(self, old: str, new: str) -> None:
        self._validate_name(new)
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            known = set(wrapper.nodes())
            if old not in known:
                raise EvidenceError(f"Unknown node: '{old}'")
            if new in known and new != old:
                raise EvidenceError(f"A node named '{new}' already exists.")
            if old == new:
                return
            full = self._snapshot_locked(wrapper)
            self._rebuild_locked(wrapper, full, rename=(old, new))

    # ------------------------------------------------------------------ edges

    def add_edge(self, parent: str, child: str) -> np.ndarray:
        """Return the child's previous flat CPT so callers can undo."""
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            known = set(wrapper.nodes())
            if parent not in known or child not in known:
                raise EvidenceError(f"Unknown endpoints: {parent}, {child}")
            if parent == child:
                raise EvidenceError("A node cannot be its own parent.")
            existing_parents = list(wrapper.parents(child))
            if parent in existing_parents:
                raise EvidenceError(f"Edge {parent}→{child} already exists.")
            # Check for cycle: would adding parent→child make parent reachable from child?
            if self._would_create_cycle(wrapper, parent, child):
                raise EvidenceError(
                    f"Edge {parent}→{child} would create a cycle."
                )
            graph = self._ensure_graph(wrapper)
            old_cpt = np.asarray(wrapper._cpts.get(child, np.array([], dtype=np.float64))).copy()
            graph.add_edge(parent, child)
            new_parents = existing_parents + [parent]
            child_states = list(wrapper.get_outcomes(child))
            self._initialise_cpt(wrapper, child, child_states, new_parents)
            wrapper._cache_metadata()
            self._invalidate(wrapper)
            return old_cpt

    def remove_edge(self, parent: str, child: str) -> np.ndarray:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            existing_parents = list(wrapper.parents(child))
            if parent not in existing_parents:
                raise EvidenceError(f"No edge {parent}→{child} exists.")
            full = self._snapshot_locked(wrapper)
            old_cpt = np.asarray(wrapper._cpts.get(child, np.array([], dtype=np.float64))).copy()
            removed_parents = {child: {parent}}
            self._rebuild_locked(wrapper, full, remove_parent_edges=removed_parents)
            return old_cpt

    # -------------------------------------------------------------------- cpt

    def set_cpt(self, node: str, shaped_2d: np.ndarray) -> np.ndarray:
        arr = np.asarray(shaped_2d, dtype=np.float64)
        if arr.ndim != 2:
            raise EvidenceError(f"CPT for '{node}' must be a 2D (rows × states) matrix.")
        row_sums = arr.sum(axis=1)
        if not np.all(np.isfinite(arr)):
            raise EvidenceError("CPT contains non-finite values.")
        if np.any(arr < -1e-12):
            raise EvidenceError("CPT values must be non-negative.")
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise EvidenceError("Each CPT row must sum to 1.0.")
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            if node not in set(wrapper.nodes()):
                raise EvidenceError(f"Unknown node: '{node}'")
            old = np.asarray(wrapper._cpts.get(node, np.array([], dtype=np.float64))).copy()
            try:
                wrapper.set_cpt(node, arr, validate=False)
            except Exception as e:  # noqa: BLE001
                raise EvidenceError(f"CPT rejected: {e}") from e
            self._invalidate(wrapper)
            return old

    def set_flat_cpt(self, node: str, flat: np.ndarray) -> None:
        """Low-level restore — used by undo to put back an old flat CPT."""
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise CompileError("No model loaded")
            graph = self._ensure_graph(wrapper)
            arr = np.ascontiguousarray(np.asarray(flat, dtype=np.float64))
            graph.set_cpt(node, arr)
            wrapper._cpts[node] = arr
            self._invalidate(wrapper)

    # ---------------------------------------------------------- rebuild core

    def _rebuild_locked(
        self,
        wrapper,
        snap: GraphSnapshot,
        *,
        exclude_node: str | None = None,
        rename: tuple[str, str] | None = None,
        remove_parent_edges: Mapping[str, set[str]] | None = None,
    ) -> None:
        rename_map: dict[str, str] = {}
        if rename is not None:
            rename_map[rename[0]] = rename[1]

        def rn(name: str) -> str:
            return rename_map.get(name, name)

        graph = Graph()
        new_order: list[str] = []
        new_states: dict[str, list[str]] = {}
        new_parents: dict[str, list[str]] = {}
        new_cpts: dict[str, np.ndarray] = {}

        for name in snap.order:
            if exclude_node is not None and name == exclude_node:
                continue
            new_name = rn(name)
            states = list(snap.states[name])
            graph.add_variable(new_name, states)
            new_order.append(new_name)
            new_states[new_name] = states

        for name in snap.order:
            if exclude_node is not None and name == exclude_node:
                continue
            new_name = rn(name)
            parents = []
            for p in snap.parents[name]:
                if exclude_node is not None and p == exclude_node:
                    continue
                if remove_parent_edges and name in remove_parent_edges:
                    if p in remove_parent_edges[name]:
                        continue
                parents.append(rn(p))
            for p in parents:
                graph.add_edge(p, new_name)
            new_parents[new_name] = parents

        # CPTs: re-initialise only when parent structure changed for this node;
        # otherwise preserve the previous flat CPT.
        for name in snap.order:
            if exclude_node is not None and name == exclude_node:
                continue
            new_name = rn(name)
            old_parents = list(snap.parents[name])
            effective_old_parents = [
                p for p in old_parents
                if not (exclude_node is not None and p == exclude_node)
                and not (remove_parent_edges and name in remove_parent_edges and p in remove_parent_edges[name])
            ]
            if [rn(p) for p in effective_old_parents] == new_parents[new_name]:
                flat = np.asarray(snap.cpts.get(name), dtype=np.float64)
                if flat.size:
                    graph.set_cpt(new_name, flat)
                    new_cpts[new_name] = flat
                else:
                    uniform = self._build_uniform(
                        new_states[new_name], [new_states[p] for p in new_parents[new_name]]
                    )
                    graph.set_cpt(new_name, uniform.ravel(order="C"))
                    new_cpts[new_name] = uniform.ravel(order="C")
            else:
                uniform = self._build_uniform(
                    new_states[new_name], [new_states[p] for p in new_parents[new_name]]
                )
                graph.set_cpt(new_name, uniform.ravel(order="C"))
                new_cpts[new_name] = uniform.ravel(order="C")

        wrapper._graph = graph
        wrapper._cpts = new_cpts
        wrapper._cache_metadata()
        # Clear runtime inference artefacts so the next query recompiles.
        wrapper._evidence = {}
        self._invalidate(wrapper)

    # -------------------------------------------------------- helpers (state)

    def _ensure_graph(self, wrapper) -> Graph:
        if wrapper._graph is None:
            wrapper._graph = Graph()
        return wrapper._graph

    def _invalidate(self, wrapper) -> None:
        wrapper._is_compiled = False
        wrapper._jt = None
        wrapper._engine = None
        self._session.invalidate_compile()

    def _snapshot_locked(self, wrapper) -> GraphSnapshot:
        order = tuple(wrapper.nodes())
        states = {n: tuple(wrapper.get_outcomes(n)) for n in order}
        parents = {n: tuple(wrapper.parents(n)) for n in order}
        cpts: dict[str, np.ndarray] = {}
        for n in order:
            raw = wrapper._cpts.get(n)
            if raw is None:
                continue
            cpts[n] = np.asarray(raw, dtype=np.float64).copy()
        return GraphSnapshot(order=order, states=states, parents=parents, cpts=cpts)

    def _node_snapshot_locked(self, wrapper, name: str) -> NodeSnapshot:
        states = tuple(wrapper.get_outcomes(name))
        parents = tuple(wrapper.parents(name))
        children = tuple(wrapper.children(name))
        flat = np.asarray(
            wrapper._cpts.get(name, np.array([], dtype=np.float64)), dtype=np.float64
        ).copy()
        if flat.size:
            try:
                shaped = flat.reshape(-1, len(states))
            except ValueError:
                shaped = flat.reshape(1, -1)
        else:
            shaped = np.zeros((0, 0))
        return NodeSnapshot(
            name=name,
            states=states,
            parents=parents,
            children=children,
            cpt_flat=flat,
            cpt_shaped=shaped,
        )

    # -------------------------------------------------------------- CPT init

    def _initialise_cpt(
        self,
        wrapper,
        node: str,
        states: Sequence[str],
        parents: Sequence[str],
    ) -> None:
        parent_state_lists = [list(wrapper.get_outcomes(p)) for p in parents]
        uniform = self._build_uniform(list(states), parent_state_lists)
        flat = np.ascontiguousarray(uniform.ravel(order="C"))
        graph = self._ensure_graph(wrapper)
        graph.set_cpt(node, flat)
        wrapper._cpts[node] = flat

    @staticmethod
    def _build_uniform(
        node_states: Sequence[str],
        parent_state_lists: Sequence[Sequence[str]],
    ) -> np.ndarray:
        n = len(node_states) or 1
        rows = 1
        for ps in parent_state_lists:
            rows *= max(1, len(ps))
        return np.full((rows, n), 1.0 / n, dtype=np.float64)

    # ------------------------------------------------------------- validation

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or not name.strip():
            raise EvidenceError("Node name cannot be empty.")
        if any(c.isspace() for c in name):
            raise EvidenceError("Node name cannot contain whitespace.")

    @staticmethod
    def _validate_states(states: Sequence[str]) -> None:
        if len(states) < 2:
            raise EvidenceError("A discrete node needs at least two states.")
        if len(set(states)) != len(states):
            raise EvidenceError("State names must be unique.")
        for s in states:
            if not s or not s.strip():
                raise EvidenceError("State names cannot be empty.")

    @staticmethod
    def _would_create_cycle(wrapper, parent: str, child: str) -> bool:
        # A cycle forms if `parent` is a descendant of `child`.
        stack = [child]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for kid in wrapper.children(cur):
                if kid == parent:
                    return True
                stack.append(kid)
        return False
