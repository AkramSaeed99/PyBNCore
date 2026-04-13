"""Tests for Loopy Belief Propagation approximate inference.

LBP should produce results close to exact inference on tree-structured
networks (where it is guaranteed to converge to the exact answer).
"""
import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph
from pybncore.loopy_bp import LoopyBPEngine


def create_chain_model():
    """A -> B -> C (tree-structured, LBP should be exact)."""
    graph = Graph()
    graph.add_variable("A", ["T", "F"])
    graph.add_variable("B", ["T", "F"])
    graph.add_variable("C", ["T", "F"])
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")

    cpts = {
        "A": np.array([0.6, 0.4]),
        "B": np.array([0.8, 0.2, 0.3, 0.7]),
        "C": np.array([0.9, 0.1, 0.2, 0.8]),
    }
    for name, cpt in cpts.items():
        graph.set_cpt(name, cpt)

    return graph, cpts


def create_diamond_model():
    """A -> B, A -> C, B -> D, C -> D (has a loop, LBP is approximate)."""
    graph = Graph()
    graph.add_variable("A", ["T", "F"])
    graph.add_variable("B", ["T", "F"])
    graph.add_variable("C", ["T", "F"])
    graph.add_variable("D", ["T", "F"])
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")

    cpts = {
        "A": np.array([0.5, 0.5]),
        "B": np.array([0.7, 0.3, 0.4, 0.6]),
        "C": np.array([0.6, 0.4, 0.2, 0.8]),
        "D": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.9]),
    }
    for name, cpt in cpts.items():
        graph.set_cpt(name, cpt)

    return graph, cpts


def test_loopy_bp_tree_exact():
    """On a tree, LBP should match exact inference."""
    graph, cpts = create_chain_model()

    # Exact inference via wrapper
    wrapper = PyBNCoreWrapper()
    wrapper._graph = graph
    wrapper._cpts = cpts
    wrapper._cache_metadata()
    wrapper._compile()
    exact = wrapper.batch_query_marginals(["A", "B", "C"])

    # LBP inference
    lbp = LoopyBPEngine(graph, cpts, damping=0.0, max_iterations=50)
    approx = lbp.infer()

    for node in ["A", "B", "C"]:
        for i, state in enumerate(wrapper._node_states[node]):
            assert np.isclose(approx[node][i], exact[node][state], atol=1e-6), \
                f"LBP mismatch on {node}[{state}]: {approx[node][i]} vs {exact[node][state]}"


def test_loopy_bp_tree_with_evidence():
    """LBP with evidence on a tree should match exact inference."""
    graph, cpts = create_chain_model()

    wrapper = PyBNCoreWrapper()
    wrapper._graph = graph
    wrapper._cpts = cpts
    wrapper._cache_metadata()
    wrapper._compile()

    wrapper.set_evidence({"A": "T"})
    exact = wrapper.batch_query_marginals(["B", "C"])

    lbp = LoopyBPEngine(graph, cpts, damping=0.0)
    approx = lbp.infer(evidence={"A": "T"})

    for node in ["B", "C"]:
        for i, state in enumerate(wrapper._node_states[node]):
            assert np.isclose(approx[node][i], exact[node][state], atol=1e-6)


def test_loopy_bp_diamond_close_to_exact():
    """On a diamond (loopy), LBP should be close to exact inference."""
    graph, cpts = create_diamond_model()

    wrapper = PyBNCoreWrapper()
    wrapper._graph = graph
    wrapper._cpts = cpts
    wrapper._cache_metadata()
    wrapper._compile()
    exact = wrapper.batch_query_marginals(["A", "B", "C", "D"])

    lbp = LoopyBPEngine(graph, cpts, damping=0.5, max_iterations=200)
    approx = lbp.infer()

    # LBP on small loopy networks should be close but not exact
    for node in ["A", "B", "C", "D"]:
        for i, state in enumerate(wrapper._node_states[node]):
            assert np.isclose(approx[node][i], exact[node][state], atol=0.05), \
                f"LBP too far from exact on {node}[{state}]: " \
                f"{approx[node][i]:.4f} vs {exact[node][state]:.4f}"


def test_loopy_bp_evidence_clamped():
    """Evidence variables should have deterministic beliefs."""
    graph, cpts = create_chain_model()
    lbp = LoopyBPEngine(graph, cpts)
    approx = lbp.infer(evidence={"A": "T"})

    assert np.isclose(approx["A"][0], 1.0)
    assert np.isclose(approx["A"][1], 0.0)


def test_loopy_bp_marginals_sum_to_one():
    """All marginals should sum to 1."""
    graph, cpts = create_diamond_model()
    lbp = LoopyBPEngine(graph, cpts)
    approx = lbp.infer()

    for node, belief in approx.items():
        assert np.isclose(belief.sum(), 1.0), \
            f"{node} marginal sums to {belief.sum()}"
