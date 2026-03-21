import pybncore as bn
import numpy as np
import time
import argparse
from generate_networks import generate_xdsl

def run_pybncore_epistemic(num_layers, nodes_per_layer, max_degree, batch_size):
    print("==================================================")
    print(" PyBNCore Epistemic (Vectorized Parameter) Profiler")
    print("==================================================")
    print(f"Generating synthetic {num_layers*nodes_per_layer}-node topology...")
    generate_xdsl("data/benchmark_epistemic.xdsl", num_layers, nodes_per_layer, max_degree)

    graph = bn.Graph()
    # Mocking standard binary nodes
    n = num_layers * nodes_per_layer
    for i in range(n):
        graph.add_variable(f"L{i // nodes_per_layer}_N{i % nodes_per_layer}", ["T", "F"])

    # Edges (re-implementing the same logic from generator)
    for layer in range(1, num_layers):
        for i in range(nodes_per_layer):
            child_idx = layer * nodes_per_layer + i
            num_parents = min(max_degree, nodes_per_layer)
            # Pick a subset of prev layer as parents
            parents = [(layer - 1) * nodes_per_layer + p for p in range(num_parents)]
            for p in parents:
                graph.add_edge(p, child_idx)

    print("Generating 50,000 batched CPT samples for all nodes...")
    for i in range(n):
        node_name = f"L{i // nodes_per_layer}_N{i % nodes_per_layer}"
        parents = graph.get_parents(i)
        fam_states = 2 ** (len(parents) + 1)
        
        # Dirichlet/Random sampling for CPTs
        samples = np.random.rand(fam_states // 2, 2, batch_size)
        # Normalize to sum to 1 over the state dimension (axis 1)
        sums = np.sum(samples, axis=1, keepdims=True)
        samples = samples / sums
        
        # Reshape to (fam_states, batch_size) flat sequence
        cpt_batched = samples.reshape(fam_states, batch_size).astype(np.float64)
        cpt_flat = np.ascontiguousarray(cpt_batched.flatten())
        graph.set_cpt(node_name, cpt_flat)

    print("[PyBNCore] Compiling structural tree...")
    jt = bn.JunctionTreeCompiler.compile(graph, "min_fill")

    print(f">>> Executing Epistemic Inference over {batch_size} networks...")
    engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=2048)

    # NO evidence -- we are purely resolving query under epistemic params
    evidence = np.full((batch_size, n), -1, dtype=np.int32)
    output = np.zeros((batch_size, 2), dtype=np.float64)

    target_node = n - 1 # L9_N9

    t0 = time.time()
    engine.evaluate(evidence, output, target_node)
    t1 = time.time()

    pybncore_time = t1 - t0
    print(f"PyBNCore NATIVE TIME: {pybncore_time:.4f} seconds!")
    print(f"Throughput: {batch_size/pybncore_time:.0f} Networks/sec")

    # Export report for bash
    with open(".pybncore_epistemic_time.txt", "w") as f:
        f.write(str(pybncore_time) + "\n" + str(batch_size))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=50000)
    args = parser.parse_args()
    
    run_pybncore_epistemic(args.layers, args.width, args.degree, args.batch_size)
