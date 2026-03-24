import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph

def test_binary_noisy_or_parity():
    # Model 1: Full explicitly defined CPT
    w_explicit = PyBNCoreWrapper()
    g1 = Graph()
    g1.add_variable("X1", ["F", "T"])
    g1.add_variable("X2", ["F", "T"])
    g1.add_variable("Y", ["F", "T"])
    g1.add_edge("X1", "Y")
    g1.add_edge("X2", "Y")
    
    w_explicit._graph = g1
    w_explicit._cpts["X1"] = np.array([0.5, 0.5])
    w_explicit._cpts["X2"] = np.array([0.5, 0.5])
    
    # L=0.1, w1=0.8, w2=0.6
    # Y=F | X1=F, X2=F -> (1-0.1)*1*1 = 0.9
    # Y=F | X1=F, X2=T -> (1-0.1)*1*0.4 = 0.36
    # Y=F | X1=T, X2=F -> (1-0.1)*0.2*1 = 0.18
    # Y=F | X1=T, X2=T -> (1-0.1)*0.2*0.4 = 0.072
    w_explicit._cpts["Y"] = np.array([
        0.9,   0.1,
        0.36,  0.64,
        0.18,  0.82,
        0.072, 0.928
    ])
    w_explicit._cache_metadata()
    for n, c in w_explicit._cpts.items(): g1.set_cpt(n, c)
    w_explicit._compile()
    
    # Model 2: Divorced Noisy-MAX generator
    w_noisy = PyBNCoreWrapper()
    g2 = Graph()
    g2.add_variable("X1", ["F", "T"])
    g2.add_variable("X2", ["F", "T"])
    
    w_noisy._graph = g2
    w_noisy._cpts["X1"] = np.array([0.5, 0.5])
    w_noisy._cpts["X2"] = np.array([0.5, 0.5])
    g2.set_cpt("X1", w_noisy._cpts["X1"])
    g2.set_cpt("X2", w_noisy._cpts["X2"])
    
    link_x1 = np.array([
        [1.0, 0.0],
        [0.2, 0.8]
    ])
    link_x2 = np.array([
        [1.0, 0.0],
        [0.4, 0.6]
    ])
    leak = np.array([0.9, 0.1])
    
    # Calling cache BEFORE add_noisy_max to track base variables
    w_noisy._cache_metadata()
    
    w_noisy.add_noisy_max(
        node="Y",
        states=["F", "T"],
        parents=["X1", "X2"],
        link_matrices={"X1": link_x1, "X2": link_x2},
        leak_probs=leak
    )
    
    w_noisy._compile()
    
    # Check that Y appears in the user-visible nodes but __Z... does not
    nodes = w_noisy.nodes()
    assert "Y" in nodes
    assert "X1" in nodes
    assert "X2" in nodes
    assert not any(n.startswith("__") for n in nodes)
    
    # Compare Marginal posteriors under no evidence
    m_exp = w_explicit.batch_query_marginals(["Y"])
    m_noi = w_noisy.batch_query_marginals(["Y"])
    
    assert np.isclose(m_exp["Y"]["F"], m_noi["Y"]["F"], atol=1e-12)
    assert np.isclose(m_exp["Y"]["T"], m_noi["Y"]["T"], atol=1e-12)

    # Compare Marginal posteriors under hard evidence
    w_explicit.set_evidence({"X1": "T"})
    w_noisy.set_evidence({"X1": "T"})
    
    m_exp2 = w_explicit.batch_query_marginals(["Y"])
    m_noi2 = w_noisy.batch_query_marginals(["Y"])
    
    assert np.isclose(m_exp2["Y"]["F"], m_noi2["Y"]["F"], atol=1e-12)
    assert np.isclose(m_exp2["Y"]["T"], m_noi2["Y"]["T"], atol=1e-12)
    
    w_explicit.set_evidence({"X1": "F", "X2": "T"})
    w_noisy.set_evidence({"X1": "F", "X2": "T"})
    
    m_exp3 = w_explicit.batch_query_marginals(["Y"])
    m_noi3 = w_noisy.batch_query_marginals(["Y"])
    
    # For X1=F, X2=T, P(Y=F) is 0.36 directly in CPT, so result must be 0.36
    assert np.isclose(m_exp3["Y"]["F"], 0.36, atol=1e-12)
    assert np.isclose(m_exp3["Y"]["F"], m_noi3["Y"]["F"], atol=1e-12)
    
def test_large_in_degree_noisy_max():
    # Create a 15-parent Noisy-MAX. If this used full CPTs, it would have 2^15 = 32,768 size table
    # Noisy-MAX should compile and evaluate extremely fast, < 1ms
    w = PyBNCoreWrapper()
    g = Graph()
    
    parents = []
    links = {}
    for i in range(15):
        p_name = f"X_{i}"
        parents.append(p_name)
        g.add_variable(p_name, ["F", "T"])
        arr = np.array([0.5, 0.5])
        w._cpts[p_name] = arr
        g.set_cpt(p_name, arr)
        links[p_name] = np.array([[1.0, 0.0], [0.9, 0.1]]) # Weak effect
        
    w._graph = g
    w._cache_metadata()
    
    # 0 leak
    leak = np.array([1.0, 0.0])
    
    w.add_noisy_max("Y", ["F", "T"], parents, links, leak)
    w._compile()
    
    m = w.batch_query_marginals(["Y"])
    
    # P(Y=F) = prod P(Zi=F), P(Zi=F) = 0.5*1.0 + 0.5*0.9 = 0.95
    # P(Y=F) = 0.95^15
    expected_f = 0.95 ** 15
    assert np.isclose(m["Y"]["F"], expected_f, atol=1e-12)
