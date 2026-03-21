import xml.etree.ElementTree as ET
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
            
    return graph, cpt_dict
