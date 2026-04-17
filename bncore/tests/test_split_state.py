"""Tests for Graph.split_state() CPT resize correctness (prerequisite for DD)."""
import numpy as np
import pytest
from pybncore._core import Graph


def test_split_state_own_cpt_no_parents():
    """Split a root variable's state. Row sum (=1) must be preserved."""
    g = Graph()
    g.add_variable("X", ["Low", "High"])
    g.set_cpt("X", np.array([0.3, 0.7]))

    g.split_state(0, 1, "Mid", "VeryHigh")

    var = g.get_variable("X")
    assert var.states == ["Low", "Mid", "VeryHigh"]
    cpt = np.asarray(var.cpt)
    assert cpt.shape == (3,)
    assert np.isclose(cpt.sum(), 1.0), f"Row sum {cpt.sum()}, expected 1.0"
    # Low unchanged, High split in half
    assert np.isclose(cpt[0], 0.3)
    assert np.isclose(cpt[1], 0.35)
    assert np.isclose(cpt[2], 0.35)


def test_split_state_own_cpt_with_parent():
    """Split X when X has a parent. All parent-conditioned rows must sum to 1."""
    g = Graph()
    g.add_variable("P", ["A", "B"])
    g.add_variable("X", ["Low", "High"])
    g.add_edge("P", "X")
    g.set_cpt("P", np.array([0.5, 0.5]))
    g.set_cpt("X", np.array([0.2, 0.8,    # P=A: p(X|A)
                              0.6, 0.4]))  # P=B: p(X|B)

    g.split_state(1, 0, "VeryLow", "Low")  # split the "Low" state

    x = g.get_variable("X")
    assert x.states == ["VeryLow", "Low", "High"]
    cpt = np.asarray(x.cpt).reshape(2, 3)
    # Each row (each parent config) must sum to 1
    assert np.allclose(cpt.sum(axis=1), 1.0), f"Row sums: {cpt.sum(axis=1)}"
    # P=A: 0.2 split → 0.1, 0.1; High=0.8 unchanged
    assert np.allclose(cpt[0], [0.1, 0.1, 0.8])
    # P=B: 0.6 split → 0.3, 0.3; High=0.4 unchanged
    assert np.allclose(cpt[1], [0.3, 0.3, 0.4])


def test_split_state_child_cpt_resize():
    """Split parent X — child Y's CPT must grow along X's axis."""
    g = Graph()
    g.add_variable("X", ["a", "b"])
    g.add_variable("Y", ["yes", "no"])
    g.add_edge("X", "Y")
    g.set_cpt("X", np.array([0.5, 0.5]))
    # Y's CPT: for each X state, p(Y|X)
    # Layout [X, Y]: [[0.9, 0.1], [0.2, 0.8]]
    g.set_cpt("Y", np.array([0.9, 0.1, 0.2, 0.8]))

    g.split_state(0, 0, "a1", "a2")

    x = g.get_variable("X")
    y = g.get_variable("Y")
    assert x.states == ["a1", "a2", "b"]
    assert y.states == ["yes", "no"]

    # X's own CPT: row sum preserved, 0.5 split in half
    xc = np.asarray(x.cpt)
    assert xc.shape == (3,)
    assert np.isclose(xc.sum(), 1.0)
    assert np.allclose(xc, [0.25, 0.25, 0.5])

    # Y's CPT: grew along X's axis (X is axis 0).  Shape [3, 2].
    # Both new states (a1, a2) inherit the conditional of original 'a'.
    yc = np.asarray(y.cpt).reshape(3, 2)
    assert np.allclose(yc[0], [0.9, 0.1]), "a1 should inherit 'a' conditional"
    assert np.allclose(yc[1], [0.9, 0.1]), "a2 should inherit 'a' conditional"
    assert np.allclose(yc[2], [0.2, 0.8]), "b conditional preserved"
    # Each row (parent config) still sums to 1
    assert np.allclose(yc.sum(axis=1), 1.0)


def test_split_state_multi_parent_child():
    """Split X when Y has two parents (X and Z). X is first parent."""
    g = Graph()
    g.add_variable("X", ["x0", "x1"])
    g.add_variable("Z", ["z0", "z1"])
    g.add_variable("Y", ["y0", "y1"])
    g.add_edge("X", "Y")
    g.add_edge("Z", "Y")
    g.set_cpt("X", np.array([0.6, 0.4]))
    g.set_cpt("Z", np.array([0.5, 0.5]))
    # Y's CPT: layout [X, Z, Y], row-major, size = 2*2*2 = 8
    # (X=0, Z=0): [0.9, 0.1]
    # (X=0, Z=1): [0.7, 0.3]
    # (X=1, Z=0): [0.3, 0.7]
    # (X=1, Z=1): [0.1, 0.9]
    g.set_cpt("Y", np.array([0.9, 0.1, 0.7, 0.3, 0.3, 0.7, 0.1, 0.9]))

    g.split_state(0, 1, "x1a", "x1b")

    y = g.get_variable("Y")
    # Y's CPT shape is now [3, 2, 2] = 12 entries
    yc = np.asarray(y.cpt).reshape(3, 2, 2)
    # X=0 rows unchanged
    assert np.allclose(yc[0, 0], [0.9, 0.1])
    assert np.allclose(yc[0, 1], [0.7, 0.3])
    # X=1 (now x1a) and X=2 (now x1b) BOTH get the old X=1 values
    assert np.allclose(yc[1, 0], [0.3, 0.7])
    assert np.allclose(yc[1, 1], [0.1, 0.9])
    assert np.allclose(yc[2, 0], [0.3, 0.7])
    assert np.allclose(yc[2, 1], [0.1, 0.9])
    # Row sums preserved
    assert np.allclose(yc.sum(axis=-1), 1.0)


def test_split_state_preserves_joint_validation():
    """After split, validate_cpts() must pass (row sums, [0,1] bounds)."""
    g = Graph()
    g.add_variable("A", ["a0", "a1", "a2"])
    g.add_variable("B", ["b0", "b1"])
    g.add_edge("A", "B")
    g.set_cpt("A", np.array([0.2, 0.3, 0.5]))
    g.set_cpt("B", np.array([0.9, 0.1,  # A=a0
                              0.5, 0.5,  # A=a1
                              0.2, 0.8]))  # A=a2

    g.split_state(0, 2, "a2_low", "a2_high")

    # validate_cpts should NOT raise
    g.validate_cpts(tolerance=1e-9)


def test_split_state_no_cpt_assigned():
    """Before set_cpt is called, split_state should just update state list."""
    g = Graph()
    g.add_variable("X", ["s0", "s1"])
    # No CPT assigned
    g.split_state(0, 0, "s0a", "s0b")
    assert g.get_variable("X").states == ["s0a", "s0b", "s1"]
