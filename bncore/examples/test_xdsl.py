import pybncore as bn
import os

def test_xdsl_parsing():
    filepath = os.path.join(os.path.dirname(__file__), "mock.xdsl")
    graph, cpts = bn.read_xdsl(filepath)
    
    print(f"Num nodes parsed: {graph.num_variables()}")
    var_a = graph.get_variable("A")
    print(f"Node A states: {var_a.states}")
    
    for name, probs in cpts.items():
        print(f"Node {name} CPT flat array shape: {probs.shape} values: {probs}")

    assert graph.num_variables() == 2
    assert "A" in cpts
    assert "B" in cpts
    assert cpts["B"].shape == (4,)
    
    print("XDSL parsing completed successfully!")

if __name__ == "__main__":
    test_xdsl_parsing()
