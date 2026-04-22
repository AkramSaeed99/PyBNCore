"""Parse, mutate, and serialise `SubModelLayout` trees.

Also provides XDSL round-trip helpers for the `<extensions><genie>` block so
that sub-model organisation survives save/load.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Mapping

from pybncore_gui.domain.submodel import ROOT_ID, SubModel, SubModelLayout

logger = logging.getLogger(__name__)


class SubModelService:
    # --------------------------------------------------------------- mutate

    def create_submodel(
        self,
        layout: SubModelLayout,
        name: str,
        parent_id: str = ROOT_ID,
        *,
        position: tuple[float, float, float, float] | None = None,
    ) -> SubModel:
        sid = self._fresh_id(layout, name)
        submodel = SubModel(
            id=sid,
            name=name.strip() or sid,
            parent_id=parent_id,
            position=position or (40.0, 40.0, 260.0, 160.0),
        )
        layout.add_submodel(submodel)
        return submodel

    def delete_submodel(self, layout: SubModelLayout, submodel_id: str) -> None:
        layout.remove_submodel(submodel_id)

    def rename_submodel(self, layout: SubModelLayout, submodel_id: str, new_name: str) -> None:
        sm = layout.submodels.get(submodel_id)
        if sm:
            sm.name = new_name.strip() or sm.name

    def move_nodes(
        self,
        layout: SubModelLayout,
        node_ids: Iterable[str],
        target_submodel_id: str,
    ) -> None:
        for nid in node_ids:
            layout.assign_node(nid, target_submodel_id)

    def move_submodel(
        self,
        layout: SubModelLayout,
        submodel_id: str,
        target_parent_id: str,
    ) -> None:
        layout.reassign_submodel(submodel_id, target_parent_id)

    def sync_nodes(self, layout: SubModelLayout, node_ids: Iterable[str]) -> None:
        """Ensure every current node has an entry; drop stale ones."""
        live = set(node_ids)
        for nid in list(layout.node_parent.keys()):
            if nid not in live:
                layout.drop_node(nid)
        for nid in live:
            layout.node_parent.setdefault(nid, ROOT_ID)

    # ------------------------------------------------------------- parsing

    def parse_from_xdsl(self, xdsl_path: str | Path) -> SubModelLayout:
        layout = SubModelLayout()
        try:
            tree = ET.parse(str(xdsl_path))
        except (ET.ParseError, FileNotFoundError) as e:
            logger.debug("No XDSL extensions available (%s)", e)
            return layout

        genie = tree.getroot().find(".//extensions/genie")
        if genie is None:
            return layout

        for node_el in genie.findall("./node"):
            nid = node_el.get("id")
            if nid:
                layout.node_parent[nid] = ROOT_ID

        for sm_el in genie.findall("./submodel"):
            self._ingest_submodel(sm_el, ROOT_ID, layout)
        return layout

    def parse_descriptions(self, xdsl_path: str | Path) -> dict[str, str]:
        """Walk the genie tree and return {node_id: description}.

        A node's description is taken from `<comment>` when available, else
        from `<name>` when the display name differs from the `id`.
        """
        out: dict[str, str] = {}
        try:
            tree = ET.parse(str(xdsl_path))
        except (ET.ParseError, FileNotFoundError):
            return out
        genie = tree.getroot().find(".//extensions/genie")
        if genie is None:
            return out

        def walk(el: ET.Element) -> None:
            for node_el in el.findall("./node"):
                nid = node_el.get("id")
                if not nid:
                    continue
                comment_el = node_el.find("comment")
                name_el = node_el.find("name")
                description = ""
                if comment_el is not None and comment_el.text:
                    description = comment_el.text.strip()
                elif name_el is not None and name_el.text:
                    display = name_el.text.strip()
                    if display and display != nid:
                        description = display
                if description:
                    out[nid] = description
            for sub_el in el.findall("./submodel"):
                walk(sub_el)

        walk(genie)
        return out

    def _ingest_submodel(
        self,
        sm_el: ET.Element,
        parent_id: str,
        layout: SubModelLayout,
    ) -> None:
        sid = sm_el.get("id")
        if not sid:
            return
        name_el = sm_el.find("name")
        name = (name_el.text.strip() if name_el is not None and name_el.text else sid)
        pos_el = sm_el.find("position")
        position = (0.0, 0.0, 220.0, 120.0)
        if pos_el is not None and pos_el.text:
            try:
                nums = [float(x) for x in pos_el.text.split()]
                if len(nums) >= 4:
                    position = (nums[0], nums[1], nums[2], nums[3])
            except ValueError:
                pass
        interior = sm_el.find("interior")
        outline = sm_el.find("outline")
        submodel = SubModel(
            id=sid,
            name=name,
            parent_id=parent_id,
            position=position,
            interior_color=self._color(interior, "#eef2ff"),
            outline_color=self._color(outline, "#3f3d9a"),
        )
        layout.add_submodel(submodel)

        for node_el in sm_el.findall("./node"):
            nid = node_el.get("id")
            if nid:
                layout.node_parent[nid] = sid
        for child_sm in sm_el.findall("./submodel"):
            self._ingest_submodel(child_sm, sid, layout)

    @staticmethod
    def _color(element: ET.Element | None, default: str) -> str:
        if element is None:
            return default
        color = element.get("color")
        if not color:
            return default
        if not color.startswith("#"):
            color = "#" + color
        return color

    # ----------------------------------------------------------- serialize

    def inject_genie_extensions(
        self,
        xdsl_path: str | Path,
        layout: SubModelLayout,
        node_positions: Mapping[str, tuple[float, float]],
        *,
        node_size: tuple[float, float] = (200.0, 92.0),
        network_name: str = "PyBNCore Network",
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
        """Rewrite the `<extensions>` block of an existing XDSL with our layout.

        Only visual metadata is written — node identity and CPT data are
        already on disk from the base `write_xdsl` call.
        """
        try:
            tree = ET.parse(str(xdsl_path))
        except (ET.ParseError, FileNotFoundError) as e:
            logger.warning("Cannot inject genie metadata (%s)", e)
            return
        root = tree.getroot()
        extensions = root.find("extensions")
        if extensions is None:
            extensions = ET.SubElement(root, "extensions")
        # Replace any prior <genie>.
        for existing in list(extensions.findall("genie")):
            extensions.remove(existing)

        genie = ET.SubElement(
            extensions,
            "genie",
            {"version": "1.0", "app": "PyBNCore GUI", "name": network_name},
        )

        descriptions = dict(descriptions or {})
        self._emit_nodes(
            genie, layout.children_node_ids(ROOT_ID), node_positions, node_size, descriptions
        )
        for sid in layout.children_submodel_ids(ROOT_ID):
            self._emit_submodel(
                genie, layout, sid, node_positions, node_size, descriptions
            )

        ET.indent(tree, space="  ")
        tree.write(str(xdsl_path), encoding="unicode", xml_declaration=True)

    def _emit_submodel(
        self,
        parent_el: ET.Element,
        layout: SubModelLayout,
        submodel_id: str,
        node_positions: Mapping[str, tuple[float, float]],
        node_size: tuple[float, float],
        descriptions: Mapping[str, str],
    ) -> None:
        sm = layout.submodels.get(submodel_id)
        if sm is None:
            return
        el = ET.SubElement(parent_el, "submodel", {"id": submodel_id})
        ET.SubElement(el, "name").text = sm.name
        ET.SubElement(el, "interior", {"color": sm.interior_color.lstrip("#")})
        ET.SubElement(el, "outline", {"color": sm.outline_color.lstrip("#")})
        ET.SubElement(el, "font", {"color": "000000", "name": "Arial", "size": "10"})
        ET.SubElement(el, "position").text = " ".join(str(int(x)) for x in sm.position)
        self._emit_nodes(
            el,
            layout.children_node_ids(submodel_id),
            node_positions,
            node_size,
            descriptions,
        )
        for child_sid in layout.children_submodel_ids(submodel_id):
            self._emit_submodel(
                el, layout, child_sid, node_positions, node_size, descriptions
            )

    @staticmethod
    def _emit_nodes(
        parent_el: ET.Element,
        node_ids: Iterable[str],
        node_positions: Mapping[str, tuple[float, float]],
        node_size: tuple[float, float],
        descriptions: Mapping[str, str],
    ) -> None:
        w, h = node_size
        for nid in node_ids:
            node_el = ET.SubElement(parent_el, "node", {"id": nid})
            display = descriptions.get(nid, nid)
            ET.SubElement(node_el, "name").text = display
            ET.SubElement(node_el, "interior", {"color": "e2e8f0"})
            ET.SubElement(node_el, "outline", {"color": "334155"})
            ET.SubElement(node_el, "font", {"color": "000000", "name": "Arial", "size": "10"})
            x, y = node_positions.get(nid, (0.0, 0.0))
            ET.SubElement(node_el, "position").text = f"{int(x)} {int(y)} {int(x + w)} {int(y + h)}"
            desc = descriptions.get(nid, "")
            if desc and desc != nid:
                ET.SubElement(node_el, "comment").text = desc

    # ----------------------------------------------------------------- util

    @staticmethod
    def _fresh_id(layout: SubModelLayout, name: str) -> str:
        base = "SUB_" + "".join(
            c if c.isalnum() or c == "_" else "_" for c in name.strip().upper()
        ) or "SUBMODEL"
        if base not in layout.submodels:
            return base
        i = 2
        while f"{base}_{i}" in layout.submodels:
            i += 1
        return f"{base}_{i}"
