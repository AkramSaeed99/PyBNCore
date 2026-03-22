"""
pybncore benchmark runner — reads the same JSONL scenario files as the SMILE binary.
Usage: python pybncore_bench_full.py <network.xdsl> <scenarios.jsonl>
Outputs: pybncore\t<n_scen>\t<total_s>\t<avg_ms>
"""
import sys
import time
import json
import pybncore

def run(net_path, scen_path):
    wrapper = pybncore.PyBNCoreWrapper(net_path)

    # Load scenarios
    scenarios = []
    with open(scen_path) as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    n_scen = len(scenarios)
    if n_scen == 0:
        print("No scenarios", file=sys.stderr)
        sys.exit(1)

    # Warmup: 5 random scenarios
    for sc in scenarios[:5]:
        try:
            wrapper.set_evidence(sc["evidence"])
            wrapper.batch_query_marginals(sc["queries"])
        except Exception:
            pass
        wrapper.clear_evidence()

    t0 = time.perf_counter()
    for sc in scenarios:
        try:
            wrapper.set_evidence(sc["evidence"])
            wrapper.batch_query_marginals(sc["queries"])
        except Exception:
            pass  # inconsistent evidence — still costs inference time
        wrapper.clear_evidence()
    t1 = time.perf_counter()

    total_s = t1 - t0
    avg_ms = total_s * 1000.0 / n_scen
    print(f"pybncore\t{n_scen}\t{total_s:.6f}\t{avg_ms:.6f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <network.xdsl> <scenarios.jsonl>", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
