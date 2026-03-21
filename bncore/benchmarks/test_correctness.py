import numpy as np
from adapter import load_xdsl_into_pgmpy
from generate_networks import generate_xdsl
import pybncore as bn
from pybncore.io import read_xdsl
from pgmpy.inference import VariableElimination
import os

def check_correctness():
    os.makedirs("data", exist_ok=True)
    filepath = "data/correctness_test.xdsl"
    print("Generating minimal correctness test network...")
    generate_xdsl(filepath, num_layers=3, nodes_per_layer=2, max_in_degree=2)
    
    print("\n--- Testing pgmpy baseline ---")
    pgmpy_model = load_xdsl_into_pgmpy(filepath)
    pgmpy_inference = VariableElimination(pgmpy_model)
    
    print("--- Testing pybncore engine ---")
    graph, cpt_dict = read_xdsl(filepath)
    jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_fill")
    engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)
    
    # Try querying ALL variables and check exactly where it diverges!
    for target_var_str in pgmpy_model.nodes():
        target_var_id = graph.get_variable(target_var_str).id
        
        evidence = np.full((1, graph.num_variables()), -1, dtype=np.int32)
        output = np.zeros((1, 2), dtype=np.float64) 
        engine.evaluate(evidence, output, target_var_id)
        cpp_probs = output[0]
        
        pgmpy_res = pgmpy_inference.query(variables=[target_var_str], evidence={})
        python_probs = pgmpy_res.values
        
        diff = np.abs(cpp_probs - python_probs).max()
        if diff > 1e-5:
            print(f"\n❌ DIVERGENCE DETECTED on Node {target_var_str}")
            print(f"pybncore: {cpp_probs}")
            print(f"pgmpy:    {python_probs}")
            print(f"Delta: {diff}")
            return
            
    print("\n✅ All nodes matched mathematically!")

if __name__ == "__main__":
    check_correctness()
