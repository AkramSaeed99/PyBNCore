import time
import os
import argparse
import numpy as np
import multiprocessing

# Core engine and generator
import pybncore as bn
from pybncore.io import read_xdsl
from generate_networks import generate_xdsl
from adapter import load_xdsl_into_pgmpy
from pgmpy.inference import VariableElimination

def run_performance_benchmarks(num_layers, nodes_per_layer, max_in_degree, batch_size):
    filepath = "data/benchmark_perf.xdsl"
    print(f"==================================================")
    print(f" PyBNCore Performance Multithreading Benchmark")
    print(f"==================================================")
    print(f"Graph Structure:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Width: {nodes_per_layer}")
    print(f"  - Total Nodes: {num_layers * nodes_per_layer}")
    print(f"  - Max In-Degree: {max_in_degree}")
    print(f"Batch Workload: {batch_size:,} Scenarios\n")
    
    os.makedirs("data", exist_ok=True)
    generate_xdsl(filepath, num_layers, nodes_per_layer, max_in_degree)
    
    # Target Node for querying
    target_var_str = f"L{num_layers-1}_N{nodes_per_layer-1}"
    
    # ---------------------------------------------------------
    # 1. pgmpy Baseline (Single Query Extrapolation)
    # ---------------------------------------------------------
    print("Loading network into pgmpy...")
    t0 = time.time()
    pgmpy_model = load_xdsl_into_pgmpy(filepath)
    pgmpy_inference = VariableElimination(pgmpy_model)
    t1 = time.time()
    print(f"pgmpy model load & init: {t1 - t0:.3f} s")
    
    print("\n[Baseline] Running pgmpy Variable Elimination (Single Query)...")
    t0 = time.time()
    _ = pgmpy_inference.query(variables=[target_var_str], evidence={})
    t1 = time.time()
    pgmpy_single_time = t1 - t0
    
    pgmpy_extrapolated_time = pgmpy_single_time * batch_size
    print(f"pgmpy single inference:  {pgmpy_single_time:.4f} s")
    print(f"pgmpy projected {batch_size:,} batch: ~{pgmpy_extrapolated_time:.2f} s")
    
    # ---------------------------------------------------------
    # 2. PyBNCore Native Engine
    # ---------------------------------------------------------
    print("\nLoading network into PyBNCore...")
    t0 = time.time()
    graph, cpt_dict = read_xdsl(filepath)
    t1 = time.time()
    print(f"pybncore graph load: {t1 - t0:.3f} s")
    
    print("Compiling Junction Tree...")
    t0 = time.time()
    jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_fill")
    t1 = time.time()
    print(f"pybncore JT Compilation: {t1 - t0:.3f} s")
    
    # Target memory prep
    target_var_id = graph.get_variable(target_var_str).id
    num_vars = graph.num_variables()
    
    # Generate random evidence matrix mapping (mostly -1 for no evidence, testing worst-case fill)
    print("\nAllocating zero-copy Batch Evidence memory...")
    evidence = np.full((batch_size, num_vars), -1, dtype=np.int32)
    output = np.zeros((batch_size, 2), dtype=np.float64)
    
    # Evaluate over different thread topologies
    system_threads = multiprocessing.cpu_count()
    thread_counts = [1, 2, 4, 8, system_threads]
    chunk_size = 2048 # Tuning parameter for L1/L2 cache locality
    
    print("\n[Benchmark] Executing PyBNCore C++ Batch Engine")
    print(f"| Threads | Chunk Size | Execution Time (s) | Speedup vs 1-Thr | Speedup vs pgmpy |")
    print(f"|---------|------------|--------------------|------------------|------------------|")
    
    base_time = None
    
    for thr in sorted(set(thread_counts)):
        engine = bn.BatchExecutionEngine(jt, num_threads=thr, chunk_size=chunk_size)
        
        # Warmup cache
        engine.evaluate(evidence[:100], output[:100], target_var_id)
        
        # Real execution
        t0 = time.time()
        engine.evaluate(evidence, output, target_var_id)
        t1 = time.time()
        
        exec_time = t1 - t0
        if thr == 1:
            base_time = exec_time
            
        speedup_mt = base_time / exec_time if base_time else 1.0
        speedup_pgmpy = pgmpy_extrapolated_time / exec_time
        
        print(f"| {thr:7d} | {chunk_size:10d} | {exec_time:18.4f} | {speedup_mt:15.2f}x | {speedup_pgmpy:15.2f}x |")

    print(f"\nBenchmark Complete. Native array memory utilized for batch output: {output.nbytes / (1024*1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=10, help="Graph layers (depth)")
    parser.add_argument("--width", type=int, default=10, help="Graph nodes per layer")
    parser.add_argument("--degree", type=int, default=3, help="Max in-degree per node")
    parser.add_argument("--batch_size", type=int, default=200_000, help="Number of scenarios to execute")
    args = parser.parse_args()
    
    run_performance_benchmarks(args.layers, args.width, args.degree, args.batch_size)
