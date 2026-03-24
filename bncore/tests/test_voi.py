import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph

def create_voi_model():
    # A -> B -> C
    # D -> B
    # A and D are independent causes of B
    wrapper = PyBNCoreWrapper()
    graph = Graph()
    graph.add_variable("A", ["T", "F"])
    graph.add_variable("D", ["T", "F"])
    graph.add_variable("B", ["T", "F"])
    graph.add_variable("C", ["T", "F"])
    
    graph.add_edge("A", "B")
    graph.add_edge("D", "B")
    graph.add_edge("B", "C")
    
    wrapper._graph = graph
    wrapper._cpts["A"] = np.array([0.5, 0.5])
    wrapper._cpts["D"] = np.array([0.5, 0.5])
    
    # B | A, D
    # A=T, D=T -> B=T (0.9)
    # A=T, D=F -> B=T (0.8)
    # A=F, D=T -> B=T (0.8)
    # A=F, D=F -> B=T (0.1)
    wrapper._cpts["B"] = np.array([
        0.9, 0.1,
        0.8, 0.2,
        0.8, 0.2,
        0.1, 0.9
    ])
    
    # C | B
    wrapper._cpts["C"] = np.array([0.9, 0.1, 0.1, 0.9])
    
    wrapper._cache_metadata()
    for name, cpt in wrapper._cpts.items(): wrapper._graph.set_cpt(name, cpt)
    wrapper._compile()
    return wrapper

def test_voi_basic():
    wrapper = create_voi_model()
    
    # VoI of observing B for query C
    # B is the direct parent of C. It should have high VoI.
    # C has states T, F.
    
    vois = wrapper.value_of_information("C", ["A", "B", "D"])
    
    # Sort them into a dict for easy lookup
    voi_dict = dict(vois)
    
    assert "B" in voi_dict
    assert "A" in voi_dict
    assert "D" in voi_dict
    
    # B should provide the most information about C since it d-separates A and D from C
    assert voi_dict["B"] > voi_dict["A"]
    assert voi_dict["B"] > voi_dict["D"]
    assert voi_dict["A"] > 0.0
    assert voi_dict["D"] > 0.0

def test_voi_d_separated():
    wrapper = create_voi_model()
    
    # If we observe B, then A and C are d-separated.
    # So VoI of A for query C given evidence B=T should be 0.
    wrapper.set_evidence({"B": "T"})
    
    vois = wrapper.value_of_information("C", ["A", "D"])
    voi_dict = dict(vois)
    
    # Because B is observed, C is independent of A and D
    assert np.isclose(voi_dict["A"], 0.0, atol=1e-7)
    assert np.isclose(voi_dict["D"], 0.0, atol=1e-7)

def test_voi_self():
    wrapper = create_voi_model()
    
    # If candidate is the query itself, it shouldn't be evaluated or if it is,
    # the wrapper specifically removes query_node from candidates if it's passed implicitly.
    # Let's pass it explicitly.
    vois = wrapper.value_of_information("C", ["C"])
    voi_dict = dict(vois)
    
    # H(C) is the entropy of C
    marginals = wrapper.batch_query_marginals(["C"])
    p_c = marginals["C"]["T"]
    h_c = - (p_c * np.log2(p_c) + (1 - p_c) * np.log2(1 - p_c))
    
    assert "C" in voi_dict
    assert np.isclose(voi_dict["C"], h_c, atol=1e-7)
