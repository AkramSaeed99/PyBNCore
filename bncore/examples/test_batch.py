import pybncore as bn
import numpy as np
import time

def test_batch():
    graph = bn.Graph()
    v1 = graph.add_variable("A", ["True", "False"])
    v2 = graph.add_variable("B", ["High", "Low"])
    graph.add_edge("A", "B")
    
    jt = bn.JunctionTreeCompiler.compile(graph)
    
    batch_size = 100000
    evidence = np.zeros((batch_size, graph.num_variables()), dtype=np.float64)
    
    engine = bn.BatchExecutionEngine(jt, num_threads=4, chunk_size=1024)
    
    print(f"Evaluating {batch_size} scenarios across threaded chunks...")
    
    t0 = time.time()
    result = np.zeros(batch_size, dtype=np.float64)
    engine.evaluate(evidence, result, v2)
    t1 = time.time()
    
    print(f"Evaluation complete in {t1 - t0:.4f} seconds.")
    print(f"Result shape: {result.shape}")
    assert result.shape == (batch_size,)

if __name__ == "__main__":
    test_batch()
