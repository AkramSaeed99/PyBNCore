"""Read-only view over the current model's structure (Phase 1)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pybncore_gui.domain.errors import QueryError
from pybncore_gui.domain.node import EdgeModel, NodeKind, NodeModel
from pybncore_gui.domain.session import ModelSession

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session

    def list_nodes(self) -> list[NodeModel]:
        with self._session.locked() as wrapper:
            if wrapper is None:
                return []
            nodes: list[NodeModel] = []
            for node_id in wrapper.nodes():
                try:
                    states = tuple(wrapper.get_outcomes(node_id))
                    parents = tuple(wrapper.parents(node_id))
                    kind = NodeKind.DISCRETE
                except Exception:  # noqa: BLE001 — continuous/unknown kind
                    states, parents, kind = (), (), NodeKind.UNKNOWN
                nodes.append(NodeModel(id=node_id, kind=kind, states=states, parents=parents))
            return nodes

    def list_hidden_relations(self) -> list[tuple[str, str]]:
        """Return (hidden_name, visible_child_name) pairs for divorcing parents
        introduced by `wrapper.add_noisy_max`. Hidden names start with `__`.
        """
        out: list[tuple[str, str]] = []
        with self._session.locked() as wrapper:
            if wrapper is None:
                return out
            all_names = list(wrapper._node_names)
            visible = set(wrapper.nodes())
            for name in all_names:
                if not name.startswith("__"):
                    continue
                for child in wrapper.children(name):
                    if child in visible:
                        out.append((name, child))
                        break
            return out

    def list_edges(self) -> list[EdgeModel]:
        with self._session.locked() as wrapper:
            if wrapper is None:
                return []
            edges: list[EdgeModel] = []
            for child in wrapper.nodes():
                try:
                    parents = wrapper.parents(child)
                except Exception:  # noqa: BLE001
                    continue
                for parent in parents:
                    edges.append(EdgeModel(parent=parent, child=child))
            return edges

    def get_outcomes(self, node: str) -> list[str]:
        with self._session.locked() as wrapper:
            if wrapper is None:
                raise QueryError("No model loaded")
            try:
                return list(wrapper.get_outcomes(node))
            except Exception as e:  # noqa: BLE001
                raise QueryError(f"Cannot read outcomes for '{node}': {e}") from e

    def get_cpt_shaped(self, node: str) -> Optional[np.ndarray]:
        with self._session.locked() as wrapper:
            if wrapper is None:
                return None
            try:
                return np.asarray(wrapper.get_cpt_shaped(node))
            except Exception:  # noqa: BLE001 — not all nodes expose a discrete CPT
                logger.debug("No shaped CPT available for %s", node)
                return None
