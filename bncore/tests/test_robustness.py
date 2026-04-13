"""Tests for numerical robustness and validation (Phase 4)."""
import pytest
import numpy as np
from pybncore._core import Graph


def test_cycle_detection_self_loop():
    """Self-loops should be rejected."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    with pytest.raises(Exception, match="Self-loop"):
        g.add_edge("A", "A")


def test_cycle_detection_direct():
    """A -> B -> A should be rejected."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    g.add_variable("B", ["T", "F"])
    g.add_edge("A", "B")
    with pytest.raises(Exception, match="[Cc]ycle"):
        g.add_edge("B", "A")


def test_cycle_detection_indirect():
    """A -> B -> C -> A should be rejected."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    g.add_variable("B", ["T", "F"])
    g.add_variable("C", ["T", "F"])
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    with pytest.raises(Exception, match="[Cc]ycle"):
        g.add_edge("C", "A")


def test_valid_dag_accepted():
    """Valid DAGs should be accepted without errors."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    g.add_variable("B", ["T", "F"])
    g.add_variable("C", ["T", "F"])
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "C")  # diamond, but still acyclic


def test_validate_cpts_valid():
    """Valid CPTs should pass validation."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    g.add_variable("B", ["T", "F"])
    g.add_edge("A", "B")
    g.set_cpt("A", np.array([0.6, 0.4]))
    g.set_cpt("B", np.array([0.8, 0.2, 0.3, 0.7]))
    g.validate_cpts()  # should not raise


def test_validate_cpts_out_of_range():
    """CPT values outside [0, 1] should fail."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    g.set_cpt("A", np.array([1.5, -0.5]))
    with pytest.raises(Exception, match="outside \\[0,1\\]"):
        g.validate_cpts()


def test_validate_cpts_not_sum_to_one():
    """CPT rows not summing to 1 should fail."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    g.set_cpt("A", np.array([0.3, 0.3]))  # sums to 0.6
    with pytest.raises(Exception, match="sums to"):
        g.validate_cpts()


def test_validate_cpts_nan():
    """NaN in CPT should fail."""
    g = Graph()
    g.add_variable("A", ["T", "F"])
    g.set_cpt("A", np.array([float('nan'), 0.5]))
    with pytest.raises(Exception, match="[Nn][Aa][Nn]|outside"):
        g.validate_cpts()


def test_strict_evidence_matching():
    """Fuzzy matching should no longer silently match wrong states."""
    from pybncore.wrapper import PyBNCoreWrapper

    wrapper = PyBNCoreWrapper()
    graph = Graph()
    graph.add_variable("A", ["State_0", "State_1"])
    wrapper._graph = graph
    wrapper._cpts = {"A": np.array([0.5, 0.5])}
    wrapper._cache_metadata()
    wrapper._graph.set_cpt("A", np.array([0.5, 0.5]))
    wrapper._compile()

    # Exact match should work
    ev = wrapper.make_evidence_matrix({"A": "State_0"}, batch_size=1)
    assert ev[0, 0] == 0

    # Fuzzy match should now fail
    with pytest.raises(ValueError, match="must match exactly"):
        wrapper.make_evidence_matrix({"A": "state0"}, batch_size=1)
