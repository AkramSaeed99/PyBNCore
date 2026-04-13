import xml.etree.ElementTree as ET
import re
import numpy as np
from ._core import Graph


def read_xdsl(filepath: str) -> tuple[Graph, dict[str, np.ndarray]]:
    """
    Parses a SMILE .xdsl file and constructs a pybncore.Graph.

    Returns:
        graph: The compiled pybncore.Graph object with structure applied.
        cpts: A dictionary mapping node 'id' to a flat numpy array of its probability parameters.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    graph = Graph()
    cpt_dict = {}

    # 1. Parse nodes and states exactly as they appear in XML order
    for node in root.findall(".//cpt"):
        node_id = node.get("id")
        states = [s.get("id") for s in node.findall("state")]
        graph.add_variable(node_id, states)

    # 2. Parse edges from parent IDs
    for node in root.findall(".//cpt"):
        child_id = node.get("id")
        parents = node.find("parents")
        if parents is not None and parents.text:
            for parent_id in parents.text.split():
                graph.add_edge(parent_id, child_id)

    # 3. Parse flat probability arrays for exactly match C++ DenseTensor mapping
    for node in root.findall(".//cpt"):
        node_id = node.get("id")
        probs_text = node.find("probabilities")
        if probs_text is not None and probs_text.text:
            probs = np.array(probs_text.text.split(), dtype=np.float64)
            cpt_dict[node_id] = probs
            graph.set_cpt(node_id, probs)

    return graph, cpt_dict


def write_xdsl(graph: Graph, cpts: dict[str, np.ndarray], filepath: str) -> None:
    """
    Writes a pybncore Graph + CPTs to SMILE XDSL format.

    Args:
        graph: The pybncore Graph object.
        cpts: Dict mapping node name to flat CPT numpy array.
        filepath: Output file path.
    """
    root = ET.Element("smile", version="1.0", id="Network1")
    nodes_el = ET.SubElement(root, "nodes")

    num_vars = graph.num_variables()
    # Collect node info in order
    node_names = []
    for i in range(num_vars):
        var = graph.get_variable(i)
        node_names.append(var.name)

    for i in range(num_vars):
        var = graph.get_variable(i)
        cpt_el = ET.SubElement(nodes_el, "cpt", id=var.name)
        for state in var.states:
            ET.SubElement(cpt_el, "state", id=state)

        parents = graph.get_parents(i)
        if parents:
            parent_names = [graph.get_variable(p).name for p in parents]
            parents_el = ET.SubElement(cpt_el, "parents")
            parents_el.text = " ".join(parent_names)

        if var.name in cpts:
            probs_el = ET.SubElement(cpt_el, "probabilities")
            probs_el.text = " ".join(f"{v:.17g}" for v in cpts[var.name])

    # Extensions placeholder (required by SMILE)
    ET.SubElement(root, "extensions")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(filepath, encoding="unicode", xml_declaration=True)


def read_bif(filepath: str) -> tuple[Graph, dict[str, np.ndarray]]:
    """
    Reads a BIF (Bayesian Interchange Format) file.

    Supports the standard BIF format used by bnlearn, Netica, and academic
    BN repositories. Handles 'network', 'variable', and 'probability' blocks.

    Returns:
        graph: The pybncore Graph object.
        cpts: Dict mapping node name to flat CPT numpy array.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    graph = Graph()
    cpt_dict = {}

    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Parse variable blocks: variable Name { type discrete [ N ] { s1, s2, ... }; }
    var_pattern = re.compile(
        r'variable\s+(\w+)\s*\{[^}]*type\s+discrete\s*\[\s*\d+\s*\]\s*\{([^}]*)\}',
        re.DOTALL
    )
    var_order = []
    for match in var_pattern.finditer(content):
        name = match.group(1)
        states_str = match.group(2)
        states = [s.strip().strip('"') for s in states_str.split(',') if s.strip()]
        graph.add_variable(name, states)
        var_order.append(name)

    # Parse probability blocks: probability ( Child | Parent1, Parent2 ) { table ... ; }
    prob_pattern = re.compile(
        r'probability\s*\(\s*(\w+)(?:\s*\|\s*([^)]*))?\s*\)\s*\{([^}]*)\}',
        re.DOTALL
    )
    for match in prob_pattern.finditer(content):
        child = match.group(1)
        parents_str = match.group(2)
        body = match.group(3)

        if parents_str:
            parents = [p.strip() for p in parents_str.split(',') if p.strip()]
            for parent in parents:
                graph.add_edge(parent, child)

        # Extract probability values from body
        # Format 1: table 0.1 0.2 0.3 0.4 ;
        table_match = re.search(r'table\s+([\d\s.eE+-]+);', body)
        if table_match:
            values = [float(v) for v in table_match.group(1).split()]
            probs = np.array(values, dtype=np.float64)
        else:
            # Format 2: (parent_state1, parent_state2) 0.1, 0.2 ;
            values = []
            row_pattern = re.compile(r'\(([^)]*)\)\s+([\d\s.,eE+-]+);')
            for row_match in row_pattern.finditer(body):
                row_values = [float(v) for v in row_match.group(2).replace(',', ' ').split()]
                values.extend(row_values)
            probs = np.array(values, dtype=np.float64)

        if len(probs) > 0:
            cpt_dict[child] = probs
            graph.set_cpt(child, probs)

    return graph, cpt_dict
