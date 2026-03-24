import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph

def create_simple_model():
    # A -> B
    wrapper = PyBNCoreWrapper()
    graph = Graph()
    graph.add_variable("A", ["True", "False"])
    graph.add_variable("B", ["High", "Low"])
    graph.add_edge("A", "B")
    
    wrapper._graph = graph
    wrapper._cpts = {
        "A": np.array([0.7, 0.3]),
        "B": np.array([
            0.9, 0.1,  # A=True
            0.2, 0.8   # A=False
        ])
    }
    wrapper._cache_metadata()
    for name, cpt in wrapper._cpts.items():
        wrapper._graph.set_cpt(name, cpt)
        
    wrapper._compile()
    return wrapper

def test_soft_evidence_scalar():
    wrapper = create_simple_model()
    
    # Baseline without evidence
    marginals = wrapper.batch_query_marginals(["A", "B"])
    p_b_high_prior = 0.7 * 0.9 + 0.3 * 0.2
    assert np.isclose(marginals["B"]["High"], p_b_high_prior)

    # Hard evidence via set_soft_evidence
    wrapper.set_soft_evidence("A", {"True": 1.0, "False": 0.0})
    marginals_hard = wrapper.batch_query_marginals(["A", "B"])
    assert np.isclose(marginals_hard["A"]["True"], 1.0)
    assert np.isclose(marginals_hard["B"]["High"], 0.9)

    # Soft evidence
    wrapper.clear_soft_evidence()
    wrapper.set_soft_evidence("A", {"True": 0.8, "False": 0.2})
    marginals_soft = wrapper.batch_query_marginals(["A", "B"])
    
    # Expected A with soft evidence: 
    # Prior A: [0.7, 0.3]
    # Likelihood A: [0.8, 0.2]
    # Unnormalized Posterior A: [0.56, 0.06]
    # Normalized Posterior A: [0.56/0.62, 0.06/0.62] = [0.9032258, 0.0967742]
    assert np.isclose(marginals_soft["A"]["True"], 0.56 / 0.62)
    assert np.isclose(marginals_soft["A"]["False"], 0.06 / 0.62)
    
    # Expected B = sum_a P(B|a) P_posterior(A=a)
    p_b_high = (0.56/0.62) * 0.9 + (0.06/0.62) * 0.2
    assert np.isclose(marginals_soft["B"]["High"], p_b_high)

def test_soft_evidence_matrix():
    wrapper = create_simple_model()
    
    # Create batch of 2 instances
    # Row 0: Soft evidence A=[0.8, 0.2], B no evidence (we can pass uniform [1,1] or not set it)
    # Row 1: Soft evidence A=[0.1, 0.9], B no evidence
    
    # We must explicitly set uniform likelihoods for B to form the full matrix, 
    # or just use set_soft_evidence_matrix on A only since the wrapper supports it per-node.
    
    matrix_A = np.array([
        [0.8, 0.2],
        [0.1, 0.9]
    ], dtype=np.float64)
    
    wrapper.set_soft_evidence_matrix("A", matrix_A)
    
    # Dummy evidence matrix to trigger batch mode B=2
    # -1 means no HARD evidence
    dummy_ev = np.full((2, len(wrapper.nodes())), -1, dtype=np.int32)
    
    marginals = wrapper.batch_query_marginals(["A", "B"], evidence_matrix=dummy_ev)
    
    res_A = marginals["A"]
    res_B = marginals["B"]
    
    assert res_A.shape == (2, 2)
    assert res_B.shape == (2, 2)
    
    # Check Row 0 (matches scalar test)
    assert np.isclose(res_A[0, 0], 0.56 / 0.62)
    assert np.isclose(res_B[0, 0], (0.56/0.62) * 0.9 + (0.06/0.62) * 0.2)
    
    # Check Row 1
    # Prior A: [0.7, 0.3], Likelihood A: [0.1, 0.9]
    # Unnormalized: [0.07, 0.27] -> sum = 0.34
    # Normalized A: [0.07/0.34, 0.27/0.34]
    p_a1_true = 0.07 / 0.34
    
    assert np.isclose(res_A[1, 0], p_a1_true)
    assert np.isclose(res_B[1, 0], p_a1_true * 0.9 + (1 - p_a1_true) * 0.2)

