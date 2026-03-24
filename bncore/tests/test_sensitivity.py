import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph

def create_sensitivity_model():
    # A -> B -> C
    wrapper = PyBNCoreWrapper()
    graph = Graph()
    graph.add_variable("A", ["T", "F"])
    graph.add_variable("B", ["T", "F"])
    graph.add_variable("C", ["T", "F"])
    
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    
    wrapper._graph = graph
    # A
    wrapper._cpts["A"] = np.array([0.5, 0.5])
    # B | A
    # A=T: B=[0.8, 0.2]
    # A=F: B=[0.2, 0.8]
    wrapper._cpts["B"] = np.array([0.8, 0.2, 0.2, 0.8])
    # C | B
    # B=T: C=[0.9, 0.1]
    # B=F: C=[0.1, 0.9]
    wrapper._cpts["C"] = np.array([0.9, 0.1, 0.1, 0.9])
    
    wrapper._cache_metadata()
    for name, cpt in wrapper._cpts.items():
        wrapper._graph.set_cpt(name, cpt)
        
    wrapper._compile()
    return wrapper

def test_sensitivity_analytic():
    wrapper = create_sensitivity_model()
    
    # We want to find sensitivity of P(C=T) with respect to P(A=T)
    # Let theta = P(A=T). Then P(A=F) = 1 - theta
    # P(B=T) = P(B=T|A=T)P(A=T) + P(B=T|A=F)P(A=F)
    #        = 0.8 * theta + 0.2 * (1 - theta)
    #        = 0.6 * theta + 0.2
    # P(C=T) = P(C=T|B=T)P(B=T) + P(C=T|B=F)P(B=F)
    #        = 0.9 * P(B=T) + 0.1 * (1 - P(B=T))
    #        = 0.8 * P(B=T) + 0.1
    #        = 0.8 * (0.6 * theta + 0.2) + 0.1
    #        = 0.48 * theta + 0.16 + 0.1
    #        = 0.48 * theta + 0.26
    # So P(C=T) should be a linear function of P(A=T) with slope 0.48

    sweep = np.linspace(0.0, 1.0, 11)
    # sensitivity(query_node, query_state, target_node, parent_config, target_state, sweep_range)
    # A has no parents, empty tuple
    res = wrapper.sensitivity(
        query_node="C", query_state="T",
        target_node="A", parent_config=(), target_state="T",
        sweep_range=sweep
    )
    
    assert np.allclose(res["theta"], sweep)
    expected_posterior = 0.48 * sweep + 0.26
    assert np.allclose(res["posterior"], expected_posterior)

def test_sensitivity_ranking():
    wrapper = create_sensitivity_model()
    
    # Rank parameters for query P(C=T)
    rankings = wrapper.sensitivity_ranking(query_node="C", query_state="T", n_top=10, epsilon=0.01)
    
    assert 1 <= len(rankings) <= 10
    
    # Check that highest sensitivity is one of the CPTs
    # Slope of P(C=T) w.r.t P(A=T) is 0.48
    # Slope of P(C=T) w.r.t P(B=T|A=T) occurs when we vary P(B=T|A=T).
    # P(B=T) = theta_b * P(A=T) + 0.2 * P(A=F) = 0.5 * theta_b + 0.1
    # P(C=T) = 0.8 * P(B=T) + 0.1 = 0.8 * (0.5 * theta_b + 0.1) + 0.1 = 0.4 * theta_b + 0.18
    # So slope is 0.40.
    
    # Slope of P(C=T) w.r.t P(C=T|B=T):
    # P(B=T) = 0.5. 
    # P(C=T) = theta_c * P(B=T) + 0.1 * P(B=F) = 0.5 * theta_c + 0.05
    # So slope is 0.50.
    
    # Thus P(C=T|B=T) should be ranked highest (slope 0.50)
    top = rankings[0]
    node, p_config, state, score = top
    assert node == "C"
    assert "T" in state or "F" in state
    assert np.isclose(score, 0.50, atol=1e-2)

def test_sensitivity_independent():
    # If A and C are independent, sensitivity is 0.
    wrapper = PyBNCoreWrapper()
    graph = Graph()
    graph.add_variable("A", ["T", "F"])
    graph.add_variable("C", ["T", "F"])
    wrapper._graph = graph
    wrapper._cpts["A"] = np.array([0.5, 0.5])
    wrapper._cpts["C"] = np.array([0.9, 0.1])
    wrapper._cache_metadata()
    for name, cpt in wrapper._cpts.items(): wrapper._graph.set_cpt(name, cpt)
    wrapper._compile()
    
    res = wrapper.sensitivity(
        query_node="C", query_state="T",
        target_node="A", parent_config=(), target_state="T",
        sweep_range=np.linspace(0, 1, 5)
    )
    # Expected slope = 0
    assert np.allclose(res["posterior"], 0.9)
