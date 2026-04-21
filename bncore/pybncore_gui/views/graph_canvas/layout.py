"""Simple layered (Sugiyama-ish) layout without external dependencies."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from pybncore_gui.domain.node import EdgeModel, NodeModel

NODE_WIDTH = 160
NODE_HEIGHT = 72
H_SPACING = 240
V_SPACING = 120


def layered_positions(
    nodes: Iterable[NodeModel],
    edges: Iterable[EdgeModel],
) -> dict[str, tuple[float, float]]:
    """Return {node_id: (x, y)} using a longest-path layering."""
    nodes = list(nodes)
    edges = list(edges)
    ids = {n.id for n in nodes}

    parents_of: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e.parent in ids and e.child in ids:
            parents_of[e.child].append(e.parent)

    layer: dict[str, int] = {}

    def _layer_of(node_id: str, visiting: set[str]) -> int:
        if node_id in layer:
            return layer[node_id]
        if node_id in visiting:
            # Cycle or malformed graph — short-circuit.
            return 0
        visiting.add(node_id)
        parents = parents_of.get(node_id, [])
        if not parents:
            value = 0
        else:
            value = 1 + max(_layer_of(p, visiting) for p in parents)
        visiting.discard(node_id)
        layer[node_id] = value
        return value

    for node in nodes:
        _layer_of(node.id, set())

    groups: dict[int, list[str]] = defaultdict(list)
    for node_id, lvl in layer.items():
        groups[lvl].append(node_id)

    positions: dict[str, tuple[float, float]] = {}
    for lvl, ids_in_layer in groups.items():
        for idx, node_id in enumerate(sorted(ids_in_layer)):
            positions[node_id] = (lvl * H_SPACING, idx * V_SPACING)
    return positions
