#!/usr/bin/env python3
"""Dynamic Discretization benchmark for PRA / reliability analysis.

Runs the canonical reliability model:
    R ~ LogNormal(log_mu, log_sigma)        (component degradation rate)
    L ~ Normal(mu_load, sigma_load)          (applied load)
    M = R + L   (functional — computed via a user-supplied CPD)

Measures:
  * wall-clock per DD iteration
  * bin count and max entropy error trajectory
  * P(R < r_crit) accuracy vs closed-form LogNormal CDF

Prints a table and writes results to results/dd/ as CSV.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pybncore._core import (
    DiscretizationManager,
    Graph,
    HybridEngine,
    HybridRunConfig,
)


def lognormal_cdf(x, log_mu, log_sigma):
    if x <= 0:
        return 0.0
    return 0.5 * (1.0 + math.erf(
        (math.log(x) - log_mu) / (log_sigma * math.sqrt(2.0))
    ))


def build_reliability_model(initial_bins=10, rare_event_mode=False,
                             threshold=None):
    """Single LogNormal root variable R (degradation rate)."""
    log_mu, log_sigma = -2.0, 0.5
    g = Graph()
    g.add_variable("R", [f"b{i}" for i in range(initial_bins)])
    dm = DiscretizationManager(max_bins_per_var=80)
    dm.register_lognormal(
        0, "R", [],
        log_mu_fn=lambda pb: log_mu,
        log_sigma_fn=lambda pb: log_sigma,
        domain_lo=1e-4, domain_hi=10.0,
        initial_bins=initial_bins, log_spaced=True,
        rare_event_mode=rare_event_mode,
    )
    if threshold is not None:
        dm.add_threshold(0, threshold)
    return g, dm, (log_mu, log_sigma)


def run_iteration_trace(max_iters=10, rare_event_mode=False, threshold=None):
    """Run DD and report per-iteration state (bins, max error, wall time)."""
    g, dm, (log_mu, log_sigma) = build_reliability_model(
        initial_bins=10, rare_event_mode=rare_event_mode, threshold=threshold)
    engine = HybridEngine(g, dm, 1)
    ev = np.full(1, -1, dtype=np.int32)
    qv = np.asarray([0], dtype=np.int64)

    trace = []
    # Run iteration-by-iteration by using max_iters=k, then k+1, etc.
    # This wastes work but is simple and gives us per-iter data.
    for k in range(1, max_iters + 1):
        g, dm, _ = build_reliability_model(
            initial_bins=10, rare_event_mode=rare_event_mode,
            threshold=threshold)
        engine = HybridEngine(g, dm, 1)
        cfg = HybridRunConfig()
        cfg.max_iters = k
        cfg.eps_entropy = 1e-12  # force full budget
        cfg.eps_kl = 1e-12

        t0 = time.perf_counter()
        result = engine.run(ev, qv, cfg)
        elapsed = time.perf_counter() - t0

        edges = np.asarray(result.edges[0])
        post = np.asarray(result.posteriors[0])

        # P(R < 0.05) estimate.
        threshold = 0.05
        p_tail = 0.0
        for j in range(len(post)):
            lo, hi = edges[j], edges[j + 1]
            if hi <= threshold:
                p_tail += post[j]
            elif lo < threshold < hi:
                p_tail += post[j] * (threshold - lo) / (hi - lo)
                break

        p_tail_exact = lognormal_cdf(threshold, log_mu, log_sigma)
        rel_err = abs(p_tail - p_tail_exact) / max(p_tail_exact, 1e-12)

        trace.append({
            "iter": k,
            "bins": len(post),
            "max_entropy_err": result.final_max_error,
            "wall_s": elapsed,
            "p_tail_dd": p_tail,
            "p_tail_exact": p_tail_exact,
            "rel_err": rel_err,
        })
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="results/dd")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("  Dynamic Discretization — LogNormal reliability R ~ LN(-2, 0.5)")
    print("  Query: P(R < 0.05)  (closed-form Phi(...) for comparison)")
    print("=" * 88)

    for label, ream, thresh in [
        ("DEFAULT (entropy-error)", False, None),
        ("RARE-EVENT MODE", True, 0.05),
    ]:
        print()
        print(f"--- {label} ---")
        trace = run_iteration_trace(args.max_iters, rare_event_mode=ream,
                                     threshold=thresh)
        hdr = f"  {'iter':>4} {'bins':>5} {'max_E':>12} {'wall_s':>10} "
        hdr += f"{'P_dd':>10} {'P_exact':>10} {'rel_err':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in trace:
            print(
                f"  {r['iter']:>4} {r['bins']:>5} {r['max_entropy_err']:>12.4e} "
                f"{r['wall_s']:>10.4f} "
                f"{r['p_tail_dd']:>10.4e} {r['p_tail_exact']:>10.4e} "
                f"{r['rel_err']:>10.2%}"
            )

        csv_name = "bench_dd_rare.csv" if ream else "bench_dd.csv"
        csv_path = out_dir / csv_name
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0].keys()),
                               lineterminator="\n")
            w.writeheader()
            w.writerows(trace)
        print(f"  → {csv_path}")


if __name__ == "__main__":
    main()
