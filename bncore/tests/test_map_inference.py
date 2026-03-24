import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph

def create_map_model():
    # A -> B
    wrapper = PyBNCoreWrapper()
    graph = Graph()
    graph.add_variable("A", ["True", "False"])
    graph.add_variable("B", ["High", "Low"])
    graph.add_edge("A", "B")
    
    wrapper._graph = graph
    # P(A=True)=0.4, P(A=False)=0.6
    # If A=True, B is High with 0.9.  If A=False, B is High with 0.1
    wrapper._cpts = {
        "A": np.array([0.4, 0.6]),
        "B": np.array([
            0.9, 0.1,  # A=True
            0.1, 0.9   # A=False
        ])
    }
    wrapper._cache_metadata()
    for name, cpt in wrapper._cpts.items():
        wrapper._graph.set_cpt(name, cpt)
        
    wrapper._compile()
    return wrapper

def test_map_inference_no_evidence():
    wrapper = create_map_model()
    
    # Joint probabilities:
    # A=T, B=H: 0.4 * 0.9 = 0.36
    # A=T, B=L: 0.4 * 0.1 = 0.04
    # A=F, B=H: 0.6 * 0.1 = 0.06
    # A=F, B=L: 0.6 * 0.9 = 0.54
    
    # MAP assignment without evidence should be A=False, B=Low
    map_res = wrapper.query_map()
    assert map_res["A"] == "False"
    assert map_res["B"] == "Low"

def test_map_inference_with_hard_evidence():
    wrapper = create_map_model()
    
    # Joint probs with B=High:
    # A=T: 0.36
    # A=F: 0.06
    # MAP should be A=True
    map_res = wrapper.query_map({"B": "High"})
    assert map_res["A"] == "True"
    assert map_res["B"] == "High"

def test_batch_map_inference():
    wrapper = create_map_model()
    
    # 2 rows of evidence
    # Row 0: B=High -> Expect A=True, B=High
    # Row 1: B=Low  -> Expect A=False, B=Low
    
    ev_matrix = wrapper.make_evidence_matrix(None, batch_size=2)
    ev_matrix[0, wrapper._name_to_id["B"]] = wrapper._node_states["B"].index("High")
    ev_matrix[1, wrapper._name_to_id["B"]] = wrapper._node_states["B"].index("Low")
    
    map_res_matrix = wrapper.batch_query_map(ev_matrix)
    
    assert map_res_matrix.shape == (2, 2)
    
    a_id = wrapper._name_to_id["A"]
    b_id = wrapper._name_to_id["B"]
    
    # Row 0
    assert map_res_matrix[0, a_id] == wrapper._node_states["A"].index("True")
    assert map_res_matrix[0, b_id] == wrapper._node_states["B"].index("High")
    
    # Row 1
    assert map_res_matrix[1, a_id] == wrapper._node_states["A"].index("False")
    assert map_res_matrix[1, b_id] == wrapper._node_states["B"].index("Low")
