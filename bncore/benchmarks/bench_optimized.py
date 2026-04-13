#!/usr/bin/env python3
"""
Comprehensive benchmark: SMILE vs PyBNCore (optimized)

Measures performance across multiple axes to validate the 6-phase
optimization plan. Tests both SMILE's sequential mode and PyBNCore's
vectorized mode with varying:

  1. Query scope (all-nodes vs single-node)  — validates barren pruning
  2. Evidence density (5%, 15%, 50%)          — validates pruning + lazy prop
  3. Batch size (1, 64, 1024, 50000)          — validates thread pool + SIMD
  4. Network size (50, 100, 200 nodes)        — validates scaling
  5. Triangulation heuristic                  — validates Phase 5

Usage:
    cd benchmarks/
    python bench_optimized.py --repeats 5 --outdir results/optimized

Requires: SMILE C++ library at deps/smile_cpp/libsmile.a
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np

# ---------------------------------------------------------------------------
# Ensure parent is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pybncore as bn
from pybncore.io import read_xdsl
from generate_networks import generate_xdsl

# ---------------------------------------------------------------------------
# SMILE binary management
# ---------------------------------------------------------------------------
DEPS_DIR = Path(__file__).resolve().parent.parent / "deps"
SMILE_LIB = DEPS_DIR / "smile_cpp" / "libsmile.a"
SMILE_INCLUDE = DEPS_DIR / "smile_cpp"
SMILE_LICENSE = DEPS_DIR / "smile_license"

def _ensure_smile_binary(bin_path: Path, src_path: Path) -> bool:
    """Compile SMILE benchmark binary. Returns True if successful."""
    if bin_path.exists() and os.access(bin_path, os.X_OK):
        return True
    if not SMILE_LIB.exists():
        print(f"  [SKIP] SMILE library not found at {SMILE_LIB}")
        return False
    cmd = [
        "clang++", "-O3", "-std=c++14",
        "-I", str(SMILE_INCLUDE),
        "-I", str(SMILE_LICENSE),
        str(src_path),
        str(SMILE_LIB),
        "-o", str(bin_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [SKIP] Failed to compile {src_path.name}:\n{res.stderr[:500]}")
        return False
    return True

def _parse_smile_time(stdout: str) -> float:
    for line in stdout.splitlines():
        if line.startswith("SMILE_TIME_SECONDS:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not parse SMILE time from:\n{stdout}")

# ---------------------------------------------------------------------------
# Evidence generation
# ---------------------------------------------------------------------------
def make_evidence(num_rows, num_vars, density, seed=42):
    """Generate random binary evidence matrix.

    Args:
        density: fraction of variables observed (0.0 to 1.0)
    Returns:
        evidence: int32 array (num_rows, num_vars), -1 = unobserved
        ev_cols: list of observed column indices
    """
    rng = np.random.default_rng(seed)
    evidence = np.full((num_rows, num_vars), -1, dtype=np.int32)
    n_ev = max(1, int(density * num_vars))
    cols = rng.choice(np.arange(num_vars), size=n_ev, replace=False)
    for c in cols:
        evidence[:, int(c)] = rng.integers(0, 2, size=num_rows, dtype=np.int32)
    return evidence, sorted(int(c) for c in cols)

def write_evidence_csv(path, evidence, ev_cols, var_names):
    """Write evidence CSV for SMILE binary."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([var_names[c] for c in ev_cols])
        for r in range(evidence.shape[0]):
            w.writerow([int(evidence[r, c]) for c in ev_cols])

# ---------------------------------------------------------------------------
# PyBNCore runners
# ---------------------------------------------------------------------------
def run_pybncore_point(engine, evidence, query_ids, offsets, repeats):
    """Single-row sequential inference (like SMILE's default mode)."""
    total_states = int(offsets[-1])
    out = np.zeros((1, total_states), dtype=np.float64)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(evidence.shape[0]):
            engine.evaluate_multi(evidence[i:i+1], out, query_ids, offsets)
        times.append(time.perf_counter() - t0)
    return times

def run_pybncore_batch(engine, evidence, query_ids, offsets, repeats):
    """Batched vectorized inference."""
    total_states = int(offsets[-1])
    out = np.zeros((evidence.shape[0], total_states), dtype=np.float64)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.evaluate_multi(evidence, out, query_ids, offsets)
        times.append(time.perf_counter() - t0)
    return times

# ---------------------------------------------------------------------------
# SMILE runners
# ---------------------------------------------------------------------------
def run_smile_sequential(bench_dir, bin_path, xdsl_path, csv_path, target, repeats):
    """Run SMILE sequential benchmark binary."""
    times = []
    for _ in range(repeats):
        res = subprocess.run(
            [str(bin_path), str(xdsl_path), str(csv_path), target],
            cwd=str(bench_dir), capture_output=True, text=True, check=True,
        )
        times.append(_parse_smile_time(res.stdout))
    return times

# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------
def bench_query_scope(graph, xdsl_path, evidence, ev_cols, var_names,
                      bench_dir, smile_bin, repeats, out_rows):
    """Compare full-query vs single-query to measure barren pruning impact."""
    num_vars = graph.num_variables()
    all_node_names = [graph.get_variable(i).name for i in range(num_vars)]

    scenarios = [
        ("1-node query", [all_node_names[-1]]),
        ("5-node query", all_node_names[-5:]),
        ("all-node query", all_node_names),
    ]

    for label, query_nodes in scenarios:
        jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_weight")
        engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)

        query_ids = np.array([graph.get_variable(n).id for n in query_nodes], dtype=np.int64)
        offsets = [0]
        for n in query_nodes:
            offsets.append(offsets[-1] + len(graph.get_variable(n).states))
        offsets = np.array(offsets, dtype=np.int64)

        py_times = run_pybncore_point(engine, evidence, query_ids, offsets, repeats)
        med = median(py_times)
        throughput = evidence.shape[0] / med
        out_rows.append({
            "scenario": "query_scope", "config": label,
            "engine": "pybncore", "mode": "point",
            "median_s": f"{med:.6f}", "throughput": f"{throughput:.0f}",
            "rows": evidence.shape[0],
        })

    # SMILE comparison (always queries 1 node)
    if smile_bin and smile_bin.exists():
        csv_path = bench_dir / "_tmp_evidence.csv"
        write_evidence_csv(csv_path, evidence, ev_cols, var_names)
        target = all_node_names[-1]
        sm_times = run_smile_sequential(bench_dir, smile_bin, xdsl_path, csv_path, target, repeats)
        med = median(sm_times)
        out_rows.append({
            "scenario": "query_scope", "config": "1-node query",
            "engine": "smile", "mode": "sequential",
            "median_s": f"{med:.6f}", "throughput": f"{evidence.shape[0]/med:.0f}",
            "rows": evidence.shape[0],
        })

def bench_evidence_density(graph, xdsl_path, var_names,
                           bench_dir, smile_bin, repeats, num_rows, out_rows):
    """Compare performance at different evidence densities."""
    num_vars = graph.num_variables()
    target_name = var_names[-1]
    query_ids = np.array([graph.get_variable(target_name).id], dtype=np.int64)
    n_states = len(graph.get_variable(target_name).states)
    offsets = np.array([0, n_states], dtype=np.int64)

    for density in [0.05, 0.15, 0.50]:
        label = f"{int(density*100)}% evidence"
        evidence, ev_cols = make_evidence(num_rows, num_vars, density, seed=42)

        jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_weight")
        engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)

        py_times = run_pybncore_point(engine, evidence, query_ids, offsets, repeats)
        med = median(py_times)
        out_rows.append({
            "scenario": "evidence_density", "config": label,
            "engine": "pybncore", "mode": "point",
            "median_s": f"{med:.6f}", "throughput": f"{num_rows/med:.0f}",
            "rows": num_rows,
        })

        if smile_bin and smile_bin.exists():
            csv_path = bench_dir / "_tmp_evidence.csv"
            write_evidence_csv(csv_path, evidence, ev_cols, var_names)
            sm_times = run_smile_sequential(bench_dir, smile_bin, xdsl_path, csv_path, target_name, repeats)
            med = median(sm_times)
            out_rows.append({
                "scenario": "evidence_density", "config": label,
                "engine": "smile", "mode": "sequential",
                "median_s": f"{med:.6f}", "throughput": f"{num_rows/med:.0f}",
                "rows": num_rows,
            })

def bench_batch_scaling(graph, xdsl_path, var_names,
                        bench_dir, smile_bin, repeats, out_rows):
    """Compare batch sizes to show thread pool + SIMD improvement."""
    num_vars = graph.num_variables()
    target_name = var_names[-1]
    query_ids = np.array([graph.get_variable(target_name).id], dtype=np.int64)
    n_states = len(graph.get_variable(target_name).states)
    offsets = np.array([0, n_states], dtype=np.int64)

    for batch_size in [1, 64, 1024, 50000]:
        evidence, ev_cols = make_evidence(batch_size, num_vars, 0.05, seed=42)

        jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_weight")

        if batch_size == 1:
            engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)
            py_times = run_pybncore_point(engine, evidence, query_ids, offsets, repeats)
        else:
            engine = bn.BatchExecutionEngine(jt, num_threads=0, chunk_size=2048)
            py_times = run_pybncore_batch(engine, evidence, query_ids, offsets, repeats)

        med = median(py_times)
        label = f"B={batch_size}"
        out_rows.append({
            "scenario": "batch_scaling", "config": label,
            "engine": "pybncore", "mode": "batch" if batch_size > 1 else "point",
            "median_s": f"{med:.6f}", "throughput": f"{batch_size/med:.0f}",
            "rows": batch_size,
        })

        if smile_bin and smile_bin.exists():
            csv_path = bench_dir / "_tmp_evidence.csv"
            write_evidence_csv(csv_path, evidence, ev_cols, var_names)
            sm_times = run_smile_sequential(bench_dir, smile_bin, xdsl_path, csv_path, target_name, repeats)
            med = median(sm_times)
            out_rows.append({
                "scenario": "batch_scaling", "config": label,
                "engine": "smile", "mode": "sequential",
                "median_s": f"{med:.6f}", "throughput": f"{batch_size/med:.0f}",
                "rows": batch_size,
            })

def bench_network_size(bench_dir, smile_bin, repeats, out_rows):
    """Compare across network sizes (50, 100, 200 nodes)."""
    data_dir = bench_dir / "data"

    for layers, width in [(5, 10), (10, 10), (10, 20)]:
        n_nodes = layers * width
        label = f"{n_nodes}-node"
        xdsl_path = data_dir / f"bench_opt_{n_nodes}n.xdsl"
        generate_xdsl(str(xdsl_path), layers, width, 3)

        graph, cpts = read_xdsl(str(xdsl_path))
        num_vars = graph.num_variables()
        var_names = [graph.get_variable(i).name for i in range(num_vars)]
        target_name = var_names[-1]
        query_ids = np.array([graph.get_variable(target_name).id], dtype=np.int64)
        n_states = len(graph.get_variable(target_name).states)
        offsets = np.array([0, n_states], dtype=np.int64)

        num_rows = 2000
        evidence, ev_cols = make_evidence(num_rows, num_vars, 0.05, seed=42)

        jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_weight")
        engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)

        py_times = run_pybncore_point(engine, evidence, query_ids, offsets, repeats)
        med = median(py_times)
        out_rows.append({
            "scenario": "network_size", "config": label,
            "engine": "pybncore", "mode": "point",
            "median_s": f"{med:.6f}", "throughput": f"{num_rows/med:.0f}",
            "rows": num_rows,
        })

        if smile_bin and smile_bin.exists():
            csv_path = bench_dir / "_tmp_evidence.csv"
            write_evidence_csv(csv_path, evidence, ev_cols, var_names)
            sm_times = run_smile_sequential(bench_dir, smile_bin, xdsl_path, csv_path, target_name, repeats)
            med = median(sm_times)
            out_rows.append({
                "scenario": "network_size", "config": label,
                "engine": "smile", "mode": "sequential",
                "median_s": f"{med:.6f}", "throughput": f"{num_rows/med:.0f}",
                "rows": num_rows,
            })

def bench_heuristics(graph, xdsl_path, var_names, repeats, out_rows):
    """Compare triangulation heuristics on compilation + inference time."""
    num_vars = graph.num_variables()
    target_name = var_names[-1]
    num_rows = 2000
    evidence, _ = make_evidence(num_rows, num_vars, 0.05, seed=42)

    for heuristic in ["min_weight", "min_fill", "min_degree", "weighted_min_fill"]:
        # Measure compilation time
        t0 = time.perf_counter()
        jt = bn.JunctionTreeCompiler.compile(graph, heuristic=heuristic)
        compile_time = time.perf_counter() - t0

        stats = jt.stats()
        engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)

        query_ids = np.array([graph.get_variable(target_name).id], dtype=np.int64)
        n_states = len(graph.get_variable(target_name).states)
        offsets = np.array([0, n_states], dtype=np.int64)

        py_times = run_pybncore_point(engine, evidence, query_ids, offsets, repeats)
        med = median(py_times)
        out_rows.append({
            "scenario": "heuristic", "config": heuristic,
            "engine": "pybncore", "mode": "point",
            "median_s": f"{med:.6f}", "throughput": f"{num_rows/med:.0f}",
            "rows": num_rows,
            "compile_s": f"{compile_time:.4f}",
            "treewidth": stats.treewidth,
            "num_cliques": stats.num_cliques,
            "total_entries": stats.total_table_entries,
        })

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive SMILE vs PyBNCore benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--point_rows", type=int, default=2000)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--outdir", type=str, default="results/optimized")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    data_dir = bench_dir / "data"
    data_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = bench_dir / args.outdir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate default network
    n_nodes = args.layers * args.width
    xdsl_path = data_dir / f"bench_opt_{n_nodes}n.xdsl"
    generate_xdsl(str(xdsl_path), args.layers, args.width, args.degree)
    graph, cpts = read_xdsl(str(xdsl_path))
    num_vars = graph.num_variables()
    var_names = [graph.get_variable(i).name for i in range(num_vars)]

    # Try to compile SMILE binary
    smile_bin = bench_dir / "benchmark_smile_vector"
    smile_src = bench_dir / "benchmark_smile_vector.cpp"
    has_smile = _ensure_smile_binary(smile_bin, smile_src)
    if not has_smile:
        smile_bin = None
        print("\n  SMILE not available — running PyBNCore-only benchmarks.\n")
    else:
        print("\n  SMILE binary ready.\n")

    # Default evidence
    evidence, ev_cols = make_evidence(args.point_rows, num_vars, 0.05, seed=args.seed)
    out_rows = []

    print("=" * 70)
    print("  BENCHMARK SUITE: SMILE vs PyBNCore (Optimized)")
    print("=" * 70)

    # 1. Query scope (barren pruning)
    print("\n[1/5] Query scope (barren node pruning) ...")
    bench_query_scope(graph, xdsl_path, evidence, ev_cols, var_names,
                      bench_dir, smile_bin, args.repeats, out_rows)

    # 2. Evidence density
    print("[2/5] Evidence density ...")
    bench_evidence_density(graph, xdsl_path, var_names,
                           bench_dir, smile_bin, args.repeats, args.point_rows, out_rows)

    # 3. Batch scaling
    print("[3/5] Batch scaling (thread pool + SIMD) ...")
    bench_batch_scaling(graph, xdsl_path, var_names,
                        bench_dir, smile_bin, args.repeats, out_rows)

    # 4. Network size
    print("[4/5] Network size scaling ...")
    bench_network_size(bench_dir, smile_bin, args.repeats, out_rows)

    # 5. Triangulation heuristics
    print("[5/5] Triangulation heuristics ...")
    bench_heuristics(graph, xdsl_path, var_names, args.repeats, out_rows)

    # --- Write results ---
    csv_path = out_dir / "results.csv"
    if out_rows:
        keys = list(out_rows[0].keys())
        # Collect all keys across all rows
        for r in out_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)

    # --- Print report ---
    report_path = out_dir / "report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("  RESULTS: SMILE vs PyBNCore (Optimized)")
    lines.append("=" * 70)
    lines.append(f"  Network: {args.layers}x{args.width} = {n_nodes} nodes")
    lines.append(f"  Repeats: {args.repeats}")
    lines.append("")

    current_scenario = None
    for r in out_rows:
        if r["scenario"] != current_scenario:
            current_scenario = r["scenario"]
            lines.append(f"\n--- {current_scenario.upper()} ---")
            if current_scenario == "heuristic":
                lines.append(f"  {'config':<22} {'engine':<10} {'median_s':>10} {'throughput':>12} "
                             f"{'treewidth':>10} {'cliques':>8} {'compile_s':>10}")
            else:
                lines.append(f"  {'config':<22} {'engine':<10} {'mode':<10} {'median_s':>10} {'throughput':>12}")

        if current_scenario == "heuristic":
            lines.append(f"  {r['config']:<22} {r['engine']:<10} {r['median_s']:>10} "
                         f"{r['throughput']:>12} {r.get('treewidth',''):>10} "
                         f"{r.get('num_cliques',''):>8} {r.get('compile_s',''):>10}")
        else:
            lines.append(f"  {r['config']:<22} {r['engine']:<10} {r['mode']:<10} "
                         f"{r['median_s']:>10} {r['throughput']:>12}")

    # Speedup summary
    lines.append("\n" + "=" * 70)
    lines.append("  SPEEDUP SUMMARY (SMILE time / PyBNCore time)")
    lines.append("=" * 70)

    for scenario in ["query_scope", "evidence_density", "batch_scaling", "network_size"]:
        scenario_rows = [r for r in out_rows if r["scenario"] == scenario]
        configs = sorted(set(r["config"] for r in scenario_rows))
        for config in configs:
            py_rows = [r for r in scenario_rows if r["config"] == config and r["engine"] == "pybncore"]
            sm_rows = [r for r in scenario_rows if r["config"] == config and r["engine"] == "smile"]
            if py_rows and sm_rows:
                py_t = float(py_rows[0]["median_s"])
                sm_t = float(sm_rows[0]["median_s"])
                speedup = sm_t / py_t if py_t > 0 else float('inf')
                winner = "PyBNCore" if speedup > 1 else "SMILE"
                lines.append(f"  {scenario:>20} | {config:<22} | {speedup:>6.2f}x  ({winner} wins)")

    report = "\n".join(lines)
    print(report)

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nResults saved to: {out_dir}/")
    print(f"  CSV: {csv_path.name}")
    print(f"  Report: {report_path.name}")

    # Cleanup temp files
    tmp = bench_dir / "_tmp_evidence.csv"
    if tmp.exists():
        tmp.unlink()

if __name__ == "__main__":
    main()
