import pybncore as bn
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

def run_correctness():
    batch_size = 5
    
    # Simple Asian-style mock logic
    print("Setting up Exact 4-Node Test Graph...")
    
    # 1. Setup PyBNcore model
    graph = bn.Graph()
    for n in ["A", "B", "C", "D"]:
        graph.add_variable(n, ["T", "F"])
    
    graph.add_edge(0, 1) # A -> B
    graph.add_edge(1, 2) # B -> C
    graph.add_edge(1, 3) # B -> D
    
    # 2. Setup PGMPY baseline model
    pgmpy_model = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C'), ('B', 'D')])
    
    # Build Empirical Samples (Batches)
    print(f"Generating random Dirichlet Probability Mappings for {batch_size} networks...")
    nodes = ["A", "B", "C", "D"]
    
    # Pre-generate matrices
    params = {}
    
    for i, node in enumerate(nodes):
        parents = graph.get_parents(i)
        fam_configs = 2 ** len(parents)
        
        # Dirichlet shape (fam_configs, 2, batch_size)
        samples = np.random.rand(fam_configs, 2, batch_size)
        sums = np.sum(samples, axis=1, keepdims=True)
        samples = samples / sums
        params[node] = samples

    print("Mapping samples into PyBNCore Native Tensor Space...")
    for i, node in enumerate(nodes):
        parents = graph.get_parents(i)
        fam_states = 2 ** (len(parents) + 1)
        samples = params[node]
        cpt_batched = samples.reshape(fam_states, batch_size).astype(np.float64)
        graph.set_cpt(node, np.ascontiguousarray(cpt_batched.flatten()))
        
    jt = bn.JunctionTreeCompiler.compile(graph, "min_fill")
    engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=32)
    
    evidence = np.full((batch_size, 4), -1, dtype=np.int32)
    output = np.zeros((batch_size, 2), dtype=np.float64)
    # Query P(D=True)
    engine.evaluate(evidence, output, 3)
    
    print("\n--- Testing PGMPY Iterative Exactness ---")
    pgmpy_probs = []
    
    for b in range(batch_size):
        # Build independent unbatched model
        cpd_A = TabularCPD('A', 2, [[params["A"][0, 0, b]], [params["A"][0, 1, b]]])
        
        # PGMPY expects probabilities formatted slightly differently: [[State0_P1, State0_P2], [State1_P1, State1_P2]]
        B_p = [[params["B"][0, 0, b], params["B"][1, 0, b]], 
               [params["B"][0, 1, b], params["B"][1, 1, b]]]
        cpd_B = TabularCPD('B', 2, B_p, evidence=['A'], evidence_card=[2])
        
        C_p = [[params["C"][0, 0, b], params["C"][1, 0, b]], 
               [params["C"][0, 1, b], params["C"][1, 1, b]]]
        cpd_C = TabularCPD('C', 2, C_p, evidence=['B'], evidence_card=[2])
        
        D_p = [[params["D"][0, 0, b], params["D"][1, 0, b]], 
               [params["D"][0, 1, b], params["D"][1, 1, b]]]
        cpd_D = TabularCPD('D', 2, D_p, evidence=['B'], evidence_card=[2])
        
        test_model = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C'), ('B', 'D')])
        test_model.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D)
        
        infer = VariableElimination(test_model)
        query = infer.query(variables=['D'])
        pgmpy_probs.append(query.values[0])
        print(f"[{b}] P(D=True) -> PGMPY {pgmpy_probs[-1]:.6f} | PyBNCore SIMD {output[b, 0]:.6f}")

    diff = np.abs(np.array(pgmpy_probs) - output[:, 0])
    max_err = np.max(diff)
    print(f"\nMAXIMUM COMPUTATIONAL ERROR: {max_err}")
    if max_err < 1e-4:
        print("✅ MATHEMATICAL EXACTNESS VALIDATED FOR PARAMETER UNCERTAINTIES!")
    else:
        print("❌ FAILED EXACTNESS VALIDATION!")

if __name__ == "__main__":
    run_correctness()
