"""Tests for barren node pruning.

Verifies that querying a subset of variables with pruning enabled
produces identical results to querying without pruning (full JT).
"""
import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph


def create_chain_model(length=10):
    """Create a chain model: X0 -> X1 -> ... -> X_{n-1}."""
    wrapper = PyBNCoreWrapper()
    graph = Graph()

    for i in range(length):
        graph.add_variable(f"X{i}", ["True", "False"])
    for i in range(length - 1):
        graph.add_edge(f"X{i}", f"X{i+1}")

    wrapper._graph = graph
    wrapper._cpts = {}
    wrapper._cpts["X0"] = np.array([0.6, 0.4])
    for i in range(1, length):
        wrapper._cpts[f"X{i}"] = np.array([0.8, 0.2, 0.3, 0.7])

    wrapper._cache_metadata()
    for name, cpt in wrapper._cpts.items():
        wrapper._graph.set_cpt(name, cpt)
    wrapper._compile()
    return wrapper


def create_tree_model():
    """Create a tree model:
         X0
        /  \\
      X1    X2
     / \\     \\
    X3  X4    X5
    """
    wrapper = PyBNCoreWrapper()
    graph = Graph()

    for i in range(6):
        graph.add_variable(f"X{i}", ["A", "B"])

    graph.add_edge("X0", "X1")
    graph.add_edge("X0", "X2")
    graph.add_edge("X1", "X3")
    graph.add_edge("X1", "X4")
    graph.add_edge("X2", "X5")

    wrapper._graph = graph
    wrapper._cpts = {
        "X0": np.array([0.5, 0.5]),
        "X1": np.array([0.7, 0.3, 0.4, 0.6]),
        "X2": np.array([0.6, 0.4, 0.2, 0.8]),
        "X3": np.array([0.9, 0.1, 0.3, 0.7]),
        "X4": np.array([0.8, 0.2, 0.1, 0.9]),
        "X5": np.array([0.5, 0.5, 0.4, 0.6]),
    }

    wrapper._cache_metadata()
    for name, cpt in wrapper._cpts.items():
        wrapper._graph.set_cpt(name, cpt)
    wrapper._compile()
    return wrapper


def _make_evidence_matrix(wrapper, evidence_dict):
    """Build a (1, num_vars) evidence matrix from a dict."""
    num_vars = len(wrapper.nodes())
    ev = np.full((1, num_vars), -1, dtype=np.int32)
    for name, state in evidence_dict.items():
        node_id = wrapper._name_to_id[name]
        state_idx = wrapper._node_states[name].index(state)
        ev[0, node_id] = state_idx
    return ev


def test_pruning_chain_query_subset():
    """Query only the first two nodes in a 10-node chain.
    Results must match a full (no-pruning) query."""
    wrapper = create_chain_model(10)

    all_nodes = [f"X{i}" for i in range(10)]
    full = wrapper.batch_query_marginals(all_nodes)
    partial = wrapper.batch_query_marginals(["X0", "X1"])

    assert np.isclose(partial["X0"]["True"], full["X0"]["True"])
    assert np.isclose(partial["X0"]["False"], full["X0"]["False"])
    assert np.isclose(partial["X1"]["True"], full["X1"]["True"])
    assert np.isclose(partial["X1"]["False"], full["X1"]["False"])


def test_pruning_chain_with_evidence():
    """Query X1 with evidence on X0 in a 10-node chain."""
    wrapper = create_chain_model(10)

    ev = _make_evidence_matrix(wrapper, {"X0": "True"})

    all_nodes = [f"X{i}" for i in range(10)]
    full = wrapper.batch_query_marginals(all_nodes, evidence_matrix=ev)
    partial = wrapper.batch_query_marginals(["X1"], evidence_matrix=ev)

    # With evidence_matrix, results are np.ndarray shape (1, n_states)
    assert np.allclose(partial["X1"], full["X1"])


def test_pruning_tree_skip_branch():
    """Query X3 with evidence on X1 -- the X2/X5 branch is barren."""
    wrapper = create_tree_model()

    ev = _make_evidence_matrix(wrapper, {"X1": "A"})

    all_nodes = [f"X{i}" for i in range(6)]
    full = wrapper.batch_query_marginals(all_nodes, evidence_matrix=ev)
    partial = wrapper.batch_query_marginals(["X3"], evidence_matrix=ev)

    assert np.allclose(partial["X3"], full["X3"])


def test_pruning_tree_multiple_queries():
    """Query X3 and X5 (spanning both branches) -- nothing should be pruned."""
    wrapper = create_tree_model()

    ev = _make_evidence_matrix(wrapper, {"X0": "A"})

    all_nodes = [f"X{i}" for i in range(6)]
    full = wrapper.batch_query_marginals(all_nodes, evidence_matrix=ev)
    partial = wrapper.batch_query_marginals(["X3", "X5"], evidence_matrix=ev)

    for name in ["X3", "X5"]:
        assert np.allclose(partial[name], full[name])


def test_pruning_batched_inference():
    """Pruning works correctly with batched (multi-row) inference."""
    wrapper = create_chain_model(10)

    num_vars = len(wrapper.nodes())
    ev_matrix = np.full((4, num_vars), -1, dtype=np.int32)
    x0_id = wrapper._name_to_id["X0"]
    ev_matrix[0, x0_id] = 0  # True
    ev_matrix[1, x0_id] = 1  # False
    ev_matrix[2, x0_id] = 0  # True
    ev_matrix[3, x0_id] = 1  # False

    all_nodes = [f"X{i}" for i in range(10)]
    full = wrapper.batch_query_marginals(all_nodes, evidence_matrix=ev_matrix)
    partial = wrapper.batch_query_marginals(["X1", "X2"], evidence_matrix=ev_matrix)

    assert np.allclose(partial["X1"], full["X1"])
    assert np.allclose(partial["X2"], full["X2"])


def test_pruning_no_evidence():
    """Pruning with no evidence still produces correct prior marginals."""
    wrapper = create_tree_model()

    full = wrapper.batch_query_marginals([f"X{i}" for i in range(6)])
    partial = wrapper.batch_query_marginals(["X3"])

    assert np.isclose(partial["X3"]["A"], full["X3"]["A"])
    assert np.isclose(partial["X3"]["B"], full["X3"]["B"])
