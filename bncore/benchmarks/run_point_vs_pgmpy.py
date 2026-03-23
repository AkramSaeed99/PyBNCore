import argparse
import csv
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from adapter import load_xdsl_into_pgmpy
from pybncore.io import read_xdsl


def _run_smile(sm_bin: Path, xdsl: Path, evidence_csv: Path, target: str, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        out = subprocess.run(
            [str(sm_bin), str(xdsl), str(evidence_csv), target],
            cwd=str(sm_bin.parent),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        m = re.search(r"SMILE_TIME_SECONDS:\s*([0-9eE+\-.]+)", out)
        if not m:
            raise RuntimeError(f"Failed to parse SMILE time from output:\n{out}")
        times.append(float(m.group(1)))
    return times


def _run_pybncore_cpp(
    py_bin: Path, xdsl: Path, evidence_csv: Path, target: str, repeats: int
) -> tuple[list[float], list[float]]:
    cold, warm = [], []
    for _ in range(repeats):
        out = subprocess.run(
            [str(py_bin), str(xdsl), str(evidence_csv), target],
            cwd=str(py_bin.parent),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        m1 = re.search(r"PYBNCORE_TIME_SECONDS_COLD:\s*([0-9eE+\-.]+)", out)
        m2 = re.search(r"PYBNCORE_TIME_SECONDS_WARM:\s*([0-9eE+\-.]+)", out)
        if not m1 or not m2:
            raise RuntimeError(f"Failed to parse pybncore C++ times from output:\n{out}")
        cold.append(float(m1.group(1)))
        warm.append(float(m2.group(1)))
    return cold, warm


def _run_pgmpy_point(
    xdsl: Path,
    target: str,
    evidence_matrix: np.ndarray,
    evidence_names: list[str],
    name_to_id: dict[str, int],
) -> float:
    model = load_xdsl_into_pgmpy(str(xdsl))
    infer = VariableElimination(model)
    state_maps = {v: model.get_cpds(v).state_names[v] for v in model.nodes()}

    # Warmup one query
    warm_evidence = {
        name: state_maps[name][int(evidence_matrix[0, name_to_id[name]])]
        for name in evidence_names
    }
    _ = infer.query(variables=[target], evidence=warm_evidence, show_progress=False)

    t0 = time.perf_counter()
    for r in range(evidence_matrix.shape[0]):
        evidence = {
            name: state_maps[name][int(evidence_matrix[r, name_to_id[name]])]
            for name in evidence_names
        }
        _ = infer.query(variables=[target], evidence=evidence, show_progress=False)
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Point-mode comparison: Smile vs pybncore C++ vs pgmpy")
    parser.add_argument("--xdsl", type=str, default="data/benchmark_perf_unified.xdsl")
    parser.add_argument("--target", type=str, default="L9_N9")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--evidence_nodes", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5, help="Repeats for Smile + pybncore C++")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    xdsl = (bench_dir / args.xdsl).resolve()
    out_root = (bench_dir / args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_root / f"point_cpp_vs_smile_pgmpy_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    sm_bin = (bench_dir / "benchmark_smile_vector").resolve()
    py_bin = (bench_dir / "benchmark_pybncore_point").resolve()
    if not sm_bin.exists():
        raise FileNotFoundError(f"Missing Smile benchmark binary: {sm_bin}")
    if not py_bin.exists():
        raise FileNotFoundError(f"Missing pybncore benchmark binary: {py_bin}")

    graph, cpts = read_xdsl(str(xdsl))
    name_to_id = {name: int(graph.get_variable(name).id) for name in cpts.keys()}
    var_names = [name for _, name in sorted((vid, name) for name, vid in name_to_id.items())]
    num_vars = graph.num_variables()

    rng = np.random.default_rng(args.seed)
    eligible = [n for n in var_names if n != args.target]
    if len(eligible) < args.evidence_nodes:
        raise ValueError("Not enough variables to choose evidence_nodes excluding target.")
    evidence_names = rng.choice(np.array(eligible), size=args.evidence_nodes, replace=False).tolist()
    evidence_ids = [name_to_id[n] for n in evidence_names]

    evidence = np.full((args.rows, num_vars), -1, dtype=np.int32)
    for vid in evidence_ids:
        evidence[:, vid] = rng.integers(0, 2, size=args.rows, dtype=np.int32)

    evidence_csv = out / "point_evidence_ex_target.csv"
    with evidence_csv.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(evidence_names)
        for r in range(args.rows):
            writer.writerow([int(evidence[r, vid]) for vid in evidence_ids])

    smile_times = _run_smile(sm_bin, xdsl, evidence_csv, args.target, args.repeats)
    py_cold, py_warm = _run_pybncore_cpp(py_bin, xdsl, evidence_csv, args.target, args.repeats)
    pgmpy_seconds = _run_pgmpy_point(xdsl, args.target, evidence, evidence_names, name_to_id)

    raw_cpp = pd.DataFrame(
        {
            "run": np.arange(1, args.repeats + 1),
            "smile_seconds": smile_times,
            "pybncore_cold_seconds": py_cold,
            "pybncore_warm_seconds": py_warm,
        }
    )
    raw_cpp.to_csv(out / "raw_cpp_runs.csv", index=False)

    summary = pd.DataFrame(
        [
            {"engine_variant": "smile_point_cpp", "seconds": float(np.median(smile_times))},
            {"engine_variant": "pybncore_point_cpp_warm", "seconds": float(np.median(py_warm))},
            {"engine_variant": "pgmpy_point_python", "seconds": float(pgmpy_seconds)},
        ]
    )
    summary["throughput_rows_per_sec"] = args.rows / summary["seconds"]
    summary.to_csv(out / "summary.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(
        summary["engine_variant"],
        summary["seconds"],
        color=["#ff7f0e", "#2ca02c", "#9467bd"],
    )
    plt.ylabel(f"Seconds ({args.rows} point rows)")
    plt.title("Point Mode: Smile vs pybncore C++ vs pgmpy")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "point_cpp_vs_smile_pgmpy.png", dpi=200)
    plt.close()

    with (out / "report.md").open("w") as f:
        f.write("# Point Mode Comparison: Smile vs pybncore C++ vs pgmpy\n\n")
        f.write(f"- Rows: {args.rows}\n")
        f.write(f"- Target: {args.target}\n")
        f.write(f"- Evidence columns: {', '.join(evidence_names)}\n")
        f.write(f"- Repeats (Smile/PyBNCore): {args.repeats}\n\n")
        f.write("## Timings\n\n")
        for _, row in summary.iterrows():
            f.write(f"- {row['engine_variant']}: `{row['seconds']:.6f}s`\n")
        f.write("\n## Artifacts\n\n")
        f.write("- `point_evidence_ex_target.csv`\n")
        f.write("- `raw_cpp_runs.csv`\n")
        f.write("- `summary.csv`\n")
        f.write("- `point_cpp_vs_smile_pgmpy.png`\n")

    print("Completed point-mode comparison.")
    print(f"Output directory: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
