#!/usr/bin/env python3
"""Clean A/B benchmark: d-sep ON vs d-sep OFF on the SAME network.

Measures the actual impact of Bayes-Ball pruning by toggling it at
runtime via the engine's set_dsep_enabled() API. Uses multiple random
seeds to ensure the result isn't cherry-picked on a single topology.

Evidence generation excludes the query node (Codex P1 fix) so that the
benchmark cannot degenerate into a trivial "query node already observed"
lookup.

Usage:
    python bench_dsep_ab.py [--repeats 3] [--seeds 5]
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pybncore as bn
from generate_networks import generate_xdsl
from pybncore.io import read_xdsl


def make_evidence_no_query(num_rows, num_vars, density, query_id, seed):
    """Generate evidence matrix that never observes the query node."""
    rng = np.random.default_rng(seed)
    evidence = np.full((num_rows, num_vars), -1, dtype=np.int32)
    candidates = np.setdiff1d(np.arange(num_vars), [query_id])
    n_ev = max(1, min(int(density * num_vars), len(candidates)))
    cols = rng.choice(candidates, size=n_ev, replace=False)
    for c in cols:
        evidence[:, int(c)] = rng.integers(0, 2, size=num_rows, dtype=np.int32)
    return evidence, [int(c) for c in cols]


def time_point_queries(engine, evidence, qids, offs, repeats=3):
    """Run point-mode queries and return median wall-clock seconds."""
    out = np.zeros((1, int(offs[-1])), dtype=np.float64)
    n = evidence.shape[0]
    times = []
    # Warmup (first calibration builds base messages)
    engine.evaluate_multi(evidence[0:1], out, qids, offs)
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(n):
            engine.evaluate_multi(evidence[i:i+1], out, qids, offs)
        times.append(time.perf_counter() - t0)
    return median(times)


def bayes_ball_stats(graph, query_id, ev_cols):
    """Run Bayes-Ball in Python to report what d-sep would prune."""
    ev_set = set(int(c) for c in ev_cols)
    top = set()
    bottom = set()
    queue = [(int(query_id), 0)]
    head = 0
    while head < len(queue):
        j, d = queue[head]
        head += 1
        if d == 0:  # from child
            if j not in ev_set:
                if j not in top:
                    top.add(j)
                    for p in graph.get_parents(j):
                        queue.append((int(p), 0))
                if j not in bottom:
                    bottom.add(j)
                    for c in graph.get_children(j):
                        queue.append((int(c), 1))
            else:
                top.add(j)
        else:  # from parent
            if j not in ev_set:
                if j not in bottom:
                    bottom.add(j)
                    for c in graph.get_children(j):
                        queue.append((int(c), 1))
            else:
                if j not in top:
                    top.add(j)
                    for p in graph.get_parents(j):
                        queue.append((int(p), 0))
    return len(top), len(ev_set & top)


def run_one(seed, density, num_rows, repeats, xdsl_path):
    """Run A/B at one (seed, density) pair. Returns dict of results."""
    # Fixed-seed network
    random.seed(seed)
    generate_xdsl(str(xdsl_path), 10, 10, 3)
    graph, _ = read_xdsl(str(xdsl_path))
    num_vars = graph.num_variables()
    query_name = graph.get_variable(num_vars - 1).name
    query_id = int(graph.get_variable(query_name).id)

    evidence, ev_cols = make_evidence_no_query(
        num_rows, num_vars, density, query_id, seed=seed + 1000
    )

    # Sanity check: query node must not be in evidence
    assert query_id not in ev_cols, f"Query {query_id} must not be observed"

    req_nodes, req_obs = bayes_ball_stats(graph, query_id, ev_cols)

    # Build engine + JT once (same for both tests)
    jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_fill")
    qids = np.array([query_id], dtype=np.int64)
    offs = np.array([0, 2], dtype=np.int64)

    # --- WITH d-sep enabled ---
    engine_on = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)
    engine_on.set_dsep_enabled(True)
    t_on = time_point_queries(engine_on, evidence, qids, offs, repeats)

    # --- WITH d-sep disabled ---
    engine_off = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)
    engine_off.set_dsep_enabled(False)
    t_off = time_point_queries(engine_off, evidence, qids, offs, repeats)

    return {
        "seed": seed,
        "density_pct": int(density * 100),
        "num_vars": num_vars,
        "evidence_nodes": len(ev_cols),
        "requisite_nodes": req_nodes,
        "requisite_obs": req_obs,
        "pruned_obs": len(ev_cols) - req_obs,
        "on_s": t_on,
        "off_s": t_off,
        "on_qps": num_rows / t_on,
        "off_qps": num_rows / t_off,
        "speedup": t_off / t_on,
    }


def main():
    parser = argparse.ArgumentParser(description="D-sep A/B benchmark")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num_rows", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default="results/dsep_ab")
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    data_dir = bench_dir / "data"
    data_dir.mkdir(exist_ok=True)
    xdsl_path = data_dir / "bench_ab.xdsl"

    out_dir = bench_dir / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    densities = [0.05, 0.15, 0.30, 0.50]
    seeds = list(range(42, 42 + args.seeds))

    all_results = []
    print("=" * 88)
    print("  D-SEPARATION A/B BENCHMARK — same network, same evidence, toggled at runtime")
    print("=" * 88)
    print(f"  Networks: 100 nodes (10x10), {args.seeds} random seeds, {args.repeats} repeats each")
    print(f"  Queries: {args.num_rows} point-mode queries per configuration")
    print(f"  Query node: always the last node, NEVER included in evidence")
    print()

    hdr = f"  {'seed':>4} {'dens':>5} {'req':>4} {'req_obs':>8} {'pruned':>7}   {'OFF (q/s)':>10} {'ON (q/s)':>10} {'speedup':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for seed in seeds:
        for density in densities:
            r = run_one(seed, density, args.num_rows, args.repeats, xdsl_path)
            all_results.append(r)
            print(f"  {r['seed']:>4} {r['density_pct']:>4}%  "
                  f"{r['requisite_nodes']:>3} {r['requisite_obs']:>5}/{r['evidence_nodes']:<2} "
                  f"{r['pruned_obs']:>6}   "
                  f"{r['off_qps']:>9,.0f} {r['on_qps']:>9,.0f} "
                  f"{r['speedup']:>7.2f}x")

    # --- Aggregate by density ---
    print()
    print("=" * 88)
    print("  AGGREGATE BY DENSITY (median across seeds)")
    print("=" * 88)
    print(f"  {'density':>8}   {'median OFF':>12} {'median ON':>12} {'median speedup':>15}")
    print("  " + "-" * 56)
    for d in densities:
        rs = [r for r in all_results if r["density_pct"] == int(d * 100)]
        med_off = median(r["off_qps"] for r in rs)
        med_on = median(r["on_qps"] for r in rs)
        med_sp = median(r["speedup"] for r in rs)
        print(f"  {int(d*100):>5}%      {med_off:>10,.0f}   {med_on:>10,.0f}   "
              f"{med_sp:>12.2f}x")

    # --- Write CSV ---
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        keys = list(all_results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(all_results)

    print()
    print(f"Results saved: {csv_path}")


if __name__ == "__main__":
    main()
