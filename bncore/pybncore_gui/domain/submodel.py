"""Sub-model metadata — purely graphical, no inference impact.

Sub-models are nested containers that organise nodes for presentation, matching
the `<extensions><genie><submodel>` structure used by SMILE / GeNIe XDSL files.

The root of the tree is represented by the empty string `""` — every node and
every user-defined sub-model has a `parent_id`, and top-level items point to
root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

ROOT_ID = ""
ROOT_NAME = "Root"


@dataclass(slots=True)
class SubModel:
    id: str
    name: str
    parent_id: str = ROOT_ID
    # (left, top, right, bottom) in scene coordinates for the container box.
    position: tuple[float, float, float, float] = (0.0, 0.0, 220.0, 120.0)
    interior_color: str = "#eef2ff"
    outline_color: str = "#3f3d9a"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "position": list(self.position),
            "interior_color": self.interior_color,
            "outline_color": self.outline_color,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SubModel":
        pos = data.get("position") or [0.0, 0.0, 220.0, 120.0]
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            parent_id=str(data.get("parent_id", ROOT_ID)),
            position=tuple(float(x) for x in pos),  # type: ignore[arg-type]
            interior_color=str(data.get("interior_color", "#eef2ff")),
            outline_color=str(data.get("outline_color", "#3f3d9a")),
        )


@dataclass(slots=True)
class SubModelLayout:
    """Full tree: all sub-models plus the node→parent mapping.

    `node_parent[node_id]` is the sub-model that directly owns the node
    (or `ROOT_ID` for top-level nodes).
    """

    submodels: dict[str, SubModel] = field(default_factory=dict)
    node_parent: dict[str, str] = field(default_factory=dict)

    # ----------------------------------------------------------- navigation

    def parent_of(self, item_id: str, *, is_submodel: bool = False) -> str:
        if is_submodel:
            sm = self.submodels.get(item_id)
            return sm.parent_id if sm else ROOT_ID
        return self.node_parent.get(item_id, ROOT_ID)

    def children_node_ids(self, submodel_id: str) -> list[str]:
        return [n for n, parent in self.node_parent.items() if parent == submodel_id]

    def children_submodel_ids(self, submodel_id: str) -> list[str]:
        return [
            sid for sid, sm in self.submodels.items() if sm.parent_id == submodel_id
        ]

    def path_to(self, submodel_id: str) -> list[str]:
        """Ancestor chain from root down to `submodel_id` (inclusive)."""
        if submodel_id == ROOT_ID or submodel_id not in self.submodels:
            return [ROOT_ID]
        path: list[str] = []
        cur = submodel_id
        while cur != ROOT_ID and cur in self.submodels:
            path.append(cur)
            cur = self.submodels[cur].parent_id
        path.append(ROOT_ID)
        path.reverse()
        return path

    def breadcrumb(self, submodel_id: str) -> list[tuple[str, str]]:
        """List of (id, display name) entries from root to `submodel_id`."""
        result: list[tuple[str, str]] = []
        for sid in self.path_to(submodel_id):
            if sid == ROOT_ID:
                result.append((ROOT_ID, ROOT_NAME))
            else:
                sm = self.submodels.get(sid)
                result.append((sid, sm.name if sm else sid))
        return result

    # ------------------------------------------------------------ mutation

    def add_submodel(self, sm: SubModel) -> None:
        self.submodels[sm.id] = sm

    def remove_submodel(self, submodel_id: str) -> None:
        """Reparent contents to the removed sub-model's parent."""
        if submodel_id not in self.submodels:
            return
        parent_id = self.submodels[submodel_id].parent_id
        # Re-home nodes
        for nid, pid in list(self.node_parent.items()):
            if pid == submodel_id:
                self.node_parent[nid] = parent_id
        # Re-home child sub-models
        for sid, sm in list(self.submodels.items()):
            if sm.parent_id == submodel_id:
                sm.parent_id = parent_id
        del self.submodels[submodel_id]

    def assign_node(self, node_id: str, submodel_id: str) -> None:
        self.node_parent[node_id] = submodel_id

    def drop_node(self, node_id: str) -> None:
        self.node_parent.pop(node_id, None)

    def reassign_submodel(self, submodel_id: str, new_parent: str) -> None:
        if submodel_id in self.submodels:
            # Cycle guard
            if new_parent == submodel_id:
                return
            cur = new_parent
            while cur in self.submodels:
                if cur == submodel_id:
                    return
                cur = self.submodels[cur].parent_id
            self.submodels[submodel_id].parent_id = new_parent

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "submodels": {sid: sm.to_dict() for sid, sm in self.submodels.items()},
            "node_parent": dict(self.node_parent),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SubModelLayout":
        submodels = {
            sid: SubModel.from_dict(d)
            for sid, d in (data.get("submodels") or {}).items()
        }
        node_parent = {
            str(k): str(v) for k, v in (data.get("node_parent") or {}).items()
        }
        return cls(submodels=submodels, node_parent=node_parent)
