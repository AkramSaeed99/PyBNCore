import time
import os
import argparse
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Core engine and generator
import pybncore as bn
from pybncore.io import read_xdsl
from generate_networks import generate_xdsl
from adapter import load_xdsl_into_pgmpy
from pgmpy.inference import VariableElimination

def plot_performance(num_layers, nodes_per_layer, max_in_degree, batch_size):
    filepath = "data/benchmark_perf.xdsl"
    print(f"Generating scaling validation data for plot...")
    os.makedirs("data", exist_ok=True)
    generate_xdsl(filepath, num_layers, nodes_per_layer, max_in_degree)
    target_var_str = f"L{num_layers-1}_N{nodes_per_layer-1}"
    
    # 1. pgmpy Baseline 
    print("Tracing pgmpy baseline metrics...")
    pgmpy_model = load_xdsl_into_pgmpy(filepath)
    pgmpy_inference = VariableElimination(pgmpy_model)
    
    t0 = time.time()
    _ = pgmpy_inference.query(variables=[target_var_str], evidence={})
    t1 = time.time()
    pgmpy_extrapolated_time = (t1 - t0) * batch_size
    
    # 2. PyBNcore Multithreading Scaling
    print("Compiling PyBNcore Junction Tree...")
    graph, cpt_dict = read_xdsl(filepath)
    jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_fill")
    target_var_id = graph.get_variable(target_var_str).id
    num_vars = graph.num_variables()
    
    evidence = np.full((batch_size, num_vars), -1, dtype=np.int32)
    output = np.zeros((batch_size, 2), dtype=np.float64)
    
    system_threads = multiprocessing.cpu_count()
    thread_counts = sorted(set([1, 2, 4, 8, system_threads]))
    chunk_size = 2048 
    
    results = []
    
    print("Scoring C++ execution topology times...")
    for thr in thread_counts:
        engine = bn.BatchExecutionEngine(jt, num_threads=thr, chunk_size=chunk_size)
        engine.evaluate(evidence[:10], output[:10], target_var_id) # warmup
        
        t0 = time.time()
        engine.evaluate(evidence, output, target_var_id)
        exec_time = time.time() - t0
        
        results.append({
            "Threads": str(thr),
            "Execution Time (s)": exec_time,
            "Type": "PyBNCore (C++)"
        })
        print(f"  - Threads: {thr} -> {exec_time:.2f} s")
        
    print("Tracing SMILE Commercial C++ Engine...")
    import subprocess
    try:
        res = subprocess.run(["./benchmark_smile"], capture_output=True, text=True, check=True)
        smile_time = None
        for line in res.stdout.split('\n'):
            if "SMILE Projected" in line:
                smile_time = float(line.split('~')[1].split('s')[0].strip())
                break
        if smile_time:
            # We add it as 1 Thread since SMILE is running unbatched single threaded
            results.append({
                "Threads": "1",
                "Execution Time (s)": smile_time,
                "Type": "SMILE (C++)"
            })
            print(f"  - SMILE C++: {smile_time:.2f} s")
    except Exception as e:
        print(f"SMILE Benchmark failed or not compiled: {e}")
        
    df = pd.DataFrame(results)
    
    # Generate Plots
    print("\nRendering Analytics Chart: benchmark_scaling.png")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df, 
        x="Threads", 
        y="Execution Time (s)", 
        hue="Type", 
        palette="viridis"
    )
    
    # Draw horizontal baseline line for pgmy
    plt.axhline(y=pgmpy_extrapolated_time, color='r', linestyle='--', linewidth=2, label=f"pgmpy Projected ({pgmpy_extrapolated_time:.1f}s)")
    
    plt.title(f"Batch Inference Scaling (Graph: {num_layers*nodes_per_layer} Nodes, Max-In: {max_in_degree})\nWorkload: {batch_size:,} Scenarios", fontsize=14, pad=15)
    plt.ylabel("Execution Time (Seconds) - Lower is Better", fontsize=12)
    plt.xlabel("PyBNCore C++ Thread Count", fontsize=12)
    
    plt.legend(loc="upper right")
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}s", 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 4), textcoords='offset points')

    plt.tight_layout()
    plt.savefig("benchmark_scaling.png", dpi=300)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=100_000)
    args = parser.parse_args()
    
    plot_performance(args.layers, args.width, args.degree, args.batch_size)
