import pytest
import numpy as np
from pybncore.wrapper import PyBNCoreWrapper
from pybncore._core import Graph

def test_equation_node():
    w = PyBNCoreWrapper()
    g = Graph()
    g.add_variable("A", ["0", "1", "2"])
    g.add_variable("B", ["0", "1", "2"])
    g.add_variable("C", ["0", "1", "2", "3", "4"])
    
    g.add_edge("A", "C")
    g.add_edge("B", "C")
    
    w._graph = g
    w._cpts["A"] = np.array([0.2, 0.3, 0.5])
    w._cpts["B"] = np.array([0.4, 0.4, 0.2])
    g.set_cpt("A", w._cpts["A"])
    g.set_cpt("B", w._cpts["B"])
    
    w._cache_metadata()
    
    # C = A + B
    def add_vars(a, b):
        return str(int(a) + int(b))
        
    w.set_equation("C", add_vars, ["A", "B"])
    w._compile()
    
    m = w.batch_query_marginals(["C"])
    
    # P(C=0) = P(A=0)*P(B=0) = 0.2 * 0.4 = 0.08
    # P(C=1) = P(A=0)*P(B=1) + P(A=1)*P(B=0) = 0.2*0.4 + 0.3*0.4 = 0.08 + 0.12 = 0.20
    # P(C=2) = P(A=0)*P(B=2) + P(A=1)*P(B=1) + P(A=2)*P(B=0) = 0.2*0.2 + 0.3*0.4 + 0.5*0.4 = 0.04 + 0.12 + 0.20 = 0.36
    # P(C=3) = P(A=1)*P(B=2) + P(A=2)*P(B=1) = 0.3*0.2 + 0.5*0.4 = 0.06 + 0.20 = 0.26
    # P(C=4) = P(A=2)*P(B=2) = 0.5*0.2 = 0.10
    
    assert np.isclose(m["C"]["0"], 0.08)
    assert np.isclose(m["C"]["1"], 0.20)
    assert np.isclose(m["C"]["2"], 0.36)
    assert np.isclose(m["C"]["3"], 0.26)
    assert np.isclose(m["C"]["4"], 0.10)
    
def test_equation_node_invalid_state():
    w = PyBNCoreWrapper()
    g = Graph()
    g.add_variable("A", ["0", "1"])
    g.add_variable("B", ["0", "1"])
    g.add_variable("C", ["0", "1"]) # Missing "2"
    
    g.add_edge("A", "C")
    g.add_edge("B", "C")
    
    w._graph = g
    w._node_names = ["A", "B", "C"]
    w._cache_metadata()
    
    def add_vars(a, b):
        return str(int(a) + int(b))
        
    with pytest.raises(ValueError, match="Equation returned '2', which is not a valid state"):
        w.set_equation("C", add_vars, ["A", "B"])
