import xml.etree.ElementTree as ET
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

def load_xdsl_into_pgmpy(filepath: str) -> DiscreteBayesianNetwork:
    """
    Parses a SMILE .xdsl file natively into a pgmpy DiscreteBayesianNetwork object
    for exact baseline mathematical validation tests.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    model = DiscreteBayesianNetwork()
    
    # 1. Parse nodes and add to model
    state_names_dict = {}
    for node in root.findall(".//cpt"):
        node_id = node.get("id")
        states = [s.get("id") for s in node.findall("state")]
        model.add_node(node_id)
        state_names_dict[node_id] = states
        
    # 2. Parse edges
    # We must add edges first before adding TabularCPDs
    parent_map = {}
    for node in root.findall(".//cpt"):
        child_id = node.get("id")
        parents = []
        parents_node = node.find("parents")
        if parents_node is not None and parents_node.text:
            parents = parents_node.text.split()
            for parent_id in parents:
                model.add_edge(parent_id, child_id)
        parent_map[child_id] = parents
                
    # 3. Create CPDs
    for node in root.findall(".//cpt"):
        node_id = node.get("id")
        parents = parent_map.get(node_id, [])
        
        cardinality = len(state_names_dict[node_id])
        evidence_card = [len(state_names_dict[p]) for p in parents] if parents else None
        evidence = parents if parents else None
        
        probs_text = node.find("probabilities")
        if probs_text is not None and probs_text.text:
            # XDSL stores flat probabilities iterating purely over parent configurations on the outer loop
            # and node states on the inner loop. 
            # E.g., P(A=0|B=0), P(A=1|B=0), P(A=0|B=1), P(A=1|B=1)
            # pgmpy expects the explicit 2D matrix shape to be (variable_card, product(evidence_card))
            flat_probs = np.array(probs_text.text.split(), dtype=np.float64)
            num_parent_configs = np.prod(evidence_card) if evidence_card else 1
            matrix = flat_probs.reshape((num_parent_configs, cardinality)).T
            
            cpd = TabularCPD(
                variable=node_id,
                variable_card=cardinality,
                values=matrix.tolist(),
                evidence=evidence,
                evidence_card=evidence_card,
                state_names={n: state_names_dict[n] for n in [node_id] + (evidence or [])}
            )
            model.add_cpds(cpd)
            
    # Validate mathematics structural integrities
    model.check_model()
    return model
    
if __name__ == "__main__":
    from sys import argv
    import os
    if len(argv) > 1 and os.path.exists(argv[1]):
        pgmpy_model = load_xdsl_into_pgmpy(argv[1])
        print(f"Successfully bridged {argv[1]} into pgmpy backend.")
        print(f"Total Nodes: {len(pgmpy_model.nodes())}")
        print(f"Total Edges: {len(pgmpy_model.edges())}")
