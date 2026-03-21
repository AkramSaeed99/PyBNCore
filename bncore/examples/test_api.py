import pybncore as bn
import numpy as np

def test_api():
    print("Testing bncore API...")
    # 1. Graph Construction
    graph = bn.Graph()
    v1 = graph.add_variable("A", ["True", "False"])
    v2 = graph.add_variable("B", ["High", "Low"])
    graph.add_edge("A", "B")
    
    print(f"Graph created with {graph.num_variables()} variables.")
    
    # 2. Compilation
    jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_fill")
    print("Graph compiled into Junction Tree successfully.")
    
    # 3. Execution Engine
    engine = bn.BatchExecutionEngine(jt, num_threads=4, chunk_size=1024)
    
    # Create a batch of 10 evidence samples
    batch_size = 10
    num_vars = graph.num_variables()
    evidence = np.full((batch_size, num_vars), -1, dtype=np.int32)
    
    # Set evidence: A is observed as "True" (state 0) in the first 5 samples
    # A is observed as "False" (state 1) in the next 5 samples
    evidence[:5, v1] = 0
    evidence[5:, v1] = 1
    
    num_states_B = 2
    output = np.zeros((batch_size, num_states_B), dtype=np.float64)
    
    # Evaluate
    engine.evaluate(evidence, output, v2)
    
    print("Inference engine executed successfully.")
    print("Output for variable B:")
    print(output)

if __name__ == "__main__":
    test_api()
