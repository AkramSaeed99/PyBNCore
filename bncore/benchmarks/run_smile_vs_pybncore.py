import argparse
import csv
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pybncore as bn
from pybncore.io import read_xdsl
from generate_networks import generate_xdsl


def _ensure_smile_binary(bin_path: Path, src_path: Path) -> None:
    if bin_path.exists() and os.access(bin_path, os.X_OK):
        return
    cmd = [
        "clang++",
        "-O3",
        "-std=c++14",
        str(src_path),
        "../../smile_cpp/libsmile.a",
        "-o",
        str(bin_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Failed to compile {src_path.name}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )


def _write_evidence_csv(
    csv_path: Path,
    evidence_matrix: np.ndarray,
    evidence_cols: list[int],
    var_names: list[str],
) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([var_names[c] for c in evidence_cols])
        for r in range(evidence_matrix.shape[0]):
            writer.writerow([int(evidence_matrix[r, c]) for c in evidence_cols])


def _make_evidence(
    num_rows: int,
    num_vars: int,
    var_names: list[str],
    seed: int,
    num_evidence_nodes: int = 5,
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    evidence = np.full((num_rows, num_vars), -1, dtype=np.int32)
    cols = rng.choice(np.arange(num_vars), size=min(num_evidence_nodes, num_vars), replace=False)
    for c in cols:
        evidence[:, int(c)] = rng.integers(0, 2, size=num_rows, dtype=np.int32)
    return evidence, [int(c) for c in cols]


def _parse_smile_time(stdout_text: str) -> float:
    for line in stdout_text.splitlines():
        if line.startswith("SMILE_TIME_SECONDS:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not parse SMILE_TIME_SECONDS from output:\n{stdout_text}")


def _run_smile_vector(
    bench_dir: Path,
    bin_path: Path,
    xdsl_path: Path,
    csv_path: Path,
    target_node: str,
    repeats: int,
) -> list[float]:
    times = []
    for _ in range(repeats):
        res = subprocess.run(
            [str(bin_path), str(xdsl_path), str(csv_path), target_node],
            cwd=str(bench_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        times.append(_parse_smile_time(res.stdout))
    return times


def _run_smile_uq(
    bench_dir: Path,
    bin_path: Path,
    xdsl_path: Path,
    uq_samples: int,
    target_node: str,
    repeats: int,
) -> list[float]:
    times = []
    for _ in range(repeats):
        res = subprocess.run(
            [str(bin_path), str(xdsl_path), str(uq_samples), target_node],
            cwd=str(bench_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        times.append(_parse_smile_time(res.stdout))
    return times


def _build_pybncore_uq_graph(xdsl_path: Path, uq_samples: int, seed: int):
    graph, cpts = read_xdsl(str(xdsl_path))
    rng = np.random.default_rng(seed)
    names = list(cpts.keys())
    name_to_id = {name: int(graph.get_variable(name).id) for name in names}
    id_to_name = {vid: name for name, vid in name_to_id.items()}
    node_names = [name for _, name in sorted((vid, name) for name, vid in name_to_id.items())]

    for node_name in node_names:
        meta = graph.get_variable(node_name)
        node_card = len(meta.states)
        parents = graph.get_parents(int(meta.id))
        fam_rows = 1
        for pid in parents:
            parent_name = id_to_name[int(pid)]
            fam_rows *= len(graph.get_variable(parent_name).states)
        samples = rng.uniform(0.1, 0.9, size=(fam_rows, node_card, uq_samples))
        samples /= np.sum(samples, axis=1, keepdims=True)
        cpt_batched = samples.reshape(fam_rows * node_card, uq_samples).astype(np.float64)
        graph.set_cpt(node_name, np.ascontiguousarray(cpt_batched.flatten(order="C")))
    return graph


def _run_pybncore_point(
    engine: bn.BatchExecutionEngine,
    evidence: np.ndarray,
    target_var_id: int,
    repeats: int,
) -> list[float]:
    out = np.zeros((1, 2), dtype=np.float64)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(evidence.shape[0]):
            engine.evaluate(evidence[i : i + 1], out, target_var_id)
        times.append(time.perf_counter() - t0)
    return times


def _run_pybncore_batch(
    engine: bn.BatchExecutionEngine,
    evidence: np.ndarray,
    target_var_id: int,
    repeats: int,
) -> list[float]:
    out = np.zeros((evidence.shape[0], 2), dtype=np.float64)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.evaluate(evidence, out, target_var_id)
        times.append(time.perf_counter() - t0)
    return times


def _run_pybncore_uq(
    xdsl_path: Path,
    target_var_name: str,
    uq_samples: int,
    repeats: int,
    chunk_size: int,
    seed: int,
) -> list[float]:
    times = []
    for rep in range(repeats):
        graph = _build_pybncore_uq_graph(xdsl_path, uq_samples, seed + rep)
        jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_fill")
        engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=chunk_size)
        target_id = graph.get_variable(target_var_name).id
        evidence = np.full((uq_samples, graph.num_variables()), -1, dtype=np.int32)
        out = np.zeros((uq_samples, 2), dtype=np.float64)
        t0 = time.perf_counter()
        engine.evaluate(evidence, out, int(target_id))
        times.append(time.perf_counter() - t0)
    return times


def _agg_rows(engine_name: str, mode: str, unit: str, rows_or_samples: int, times: list[float]) -> list[dict]:
    med = median(times)
    p95 = float(np.percentile(np.array(times, dtype=np.float64), 95))
    if unit == "rows":
        throughput = rows_or_samples / med
    else:
        throughput = rows_or_samples / med
    return [
        {
            "engine": engine_name,
            "mode": mode,
            "median_seconds": med,
            "p95_seconds": p95,
            "throughput_per_sec": throughput,
            "latency_ms_per_item": (med / rows_or_samples) * 1000.0,
            "count": rows_or_samples,
        }
    ]


def _plot_results(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    point_df = df[df["mode"] == "point"]
    axes[0].bar(point_df["engine"], point_df["latency_ms_per_item"], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("Point Mode Latency")
    axes[0].set_ylabel("ms / scenario")
    axes[0].set_xlabel("Engine")
    axes[0].grid(axis="y", alpha=0.3)

    thr_df = df[df["mode"].isin(["batch", "uq"])].copy()
    x_labels = sorted(thr_df["mode"].unique())
    x = np.arange(len(x_labels))
    width = 0.35
    py_vals = [float(thr_df[(thr_df["mode"] == m) & (thr_df["engine"] == "pybncore")]["throughput_per_sec"]) for m in x_labels]
    sm_vals = [float(thr_df[(thr_df["mode"] == m) & (thr_df["engine"] == "smile")]["throughput_per_sec"]) for m in x_labels]
    axes[1].bar(x - width / 2, py_vals, width, label="pybncore", color="#1f77b4")
    axes[1].bar(x + width / 2, sm_vals, width, label="smile", color="#ff7f0e")
    axes[1].set_xticks(x, x_labels)
    axes[1].set_title("Throughput (Higher is Better)")
    axes[1].set_ylabel("items / sec")
    axes[1].set_xlabel("Mode")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Smile vs pybncore benchmark runner.")
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--point_rows", type=int, default=5000)
    parser.add_argument("--batch_rows", type=int, default=50000)
    parser.add_argument("--uq_samples", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence_nodes", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    data_dir = bench_dir / "data"
    out_root = bench_dir / args.outdir
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"smile_vs_pybncore_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    xdsl_path = data_dir / "benchmark_perf_unified.xdsl"
    generate_xdsl(str(xdsl_path), args.layers, args.width, args.degree)
    target_var_name = f"L{args.layers-1}_N{args.width-1}"

    # Build pybncore benchmark objects once for point/batch.
    graph, cpts = read_xdsl(str(xdsl_path))
    jt = bn.JunctionTreeCompiler.compile(graph, heuristic="min_fill")
    target_var_id = int(graph.get_variable(target_var_name).id)
    num_vars = graph.num_variables()
    if len(cpts) != num_vars:
        raise RuntimeError(
            f"read_xdsl metadata mismatch: num_vars={num_vars} but CPT entries={len(cpts)}"
        )
    name_to_id = {name: int(graph.get_variable(name).id) for name in cpts.keys()}
    var_names = [name for _, name in sorted((vid, name) for name, vid in name_to_id.items())]

    point_evidence, point_cols = _make_evidence(
        args.point_rows, num_vars, var_names, seed=args.seed, num_evidence_nodes=args.evidence_nodes
    )
    batch_evidence, batch_cols = _make_evidence(
        args.batch_rows, num_vars, var_names, seed=args.seed + 1, num_evidence_nodes=args.evidence_nodes
    )
    point_csv = out_dir / "point_evidence.csv"
    batch_csv = out_dir / "batch_evidence.csv"
    _write_evidence_csv(point_csv, point_evidence, point_cols, var_names)
    _write_evidence_csv(batch_csv, batch_evidence, batch_cols, var_names)

    # Ensure smile binaries exist.
    smile_vec_bin = bench_dir / "benchmark_smile_vector"
    smile_vec_src = bench_dir / "benchmark_smile_vector.cpp"
    smile_uq_bin = bench_dir / "benchmark_smile_epistemic"
    smile_uq_src = bench_dir / "benchmark_smile_epistemic.cpp"
    _ensure_smile_binary(smile_vec_bin, smile_vec_src)
    _ensure_smile_binary(smile_uq_bin, smile_uq_src)

    # pybncore runs
    py_point_engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=1)
    py_batch_engine = bn.BatchExecutionEngine(jt, num_threads=1, chunk_size=args.chunk_size)
    py_point_times = _run_pybncore_point(py_point_engine, point_evidence, target_var_id, args.repeats)
    py_batch_times = _run_pybncore_batch(py_batch_engine, batch_evidence, target_var_id, args.repeats)
    py_uq_times = _run_pybncore_uq(
        xdsl_path=xdsl_path,
        target_var_name=target_var_name,
        uq_samples=args.uq_samples,
        repeats=args.repeats,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    # smile runs
    sm_point_times = _run_smile_vector(
        bench_dir=bench_dir,
        bin_path=smile_vec_bin,
        xdsl_path=xdsl_path,
        csv_path=point_csv,
        target_node=target_var_name,
        repeats=args.repeats,
    )
    sm_batch_times = _run_smile_vector(
        bench_dir=bench_dir,
        bin_path=smile_vec_bin,
        xdsl_path=xdsl_path,
        csv_path=batch_csv,
        target_node=target_var_name,
        repeats=args.repeats,
    )
    sm_uq_times = _run_smile_uq(
        bench_dir=bench_dir,
        bin_path=smile_uq_bin,
        xdsl_path=xdsl_path,
        uq_samples=args.uq_samples,
        target_node=target_var_name,
        repeats=args.repeats,
    )

    raw_rows = []
    for mode, engine_name, times in [
        ("point", "pybncore", py_point_times),
        ("batch", "pybncore", py_batch_times),
        ("uq", "pybncore", py_uq_times),
        ("point", "smile", sm_point_times),
        ("batch", "smile", sm_batch_times),
        ("uq", "smile", sm_uq_times),
    ]:
        for i, t in enumerate(times):
            raw_rows.append({"engine": engine_name, "mode": mode, "run": i + 1, "seconds": t})

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = out_dir / "raw_runs.csv"
    raw_df.to_csv(raw_csv, index=False)

    agg_rows = []
    agg_rows += _agg_rows("pybncore", "point", "rows", args.point_rows, py_point_times)
    agg_rows += _agg_rows("pybncore", "batch", "rows", args.batch_rows, py_batch_times)
    agg_rows += _agg_rows("pybncore", "uq", "samples", args.uq_samples, py_uq_times)
    agg_rows += _agg_rows("smile", "point", "rows", args.point_rows, sm_point_times)
    agg_rows += _agg_rows("smile", "batch", "rows", args.batch_rows, sm_batch_times)
    agg_rows += _agg_rows("smile", "uq", "samples", args.uq_samples, sm_uq_times)
    agg_df = pd.DataFrame(agg_rows)

    # Speedups (smile / pybncore) per mode
    speedups = {}
    for mode in ["point", "batch", "uq"]:
        sm = float(agg_df[(agg_df["mode"] == mode) & (agg_df["engine"] == "smile")]["median_seconds"])
        py = float(agg_df[(agg_df["mode"] == mode) & (agg_df["engine"] == "pybncore")]["median_seconds"])
        speedups[mode] = sm / py
    agg_df["speedup_smile_over_pybncore"] = np.nan
    for mode in ["point", "batch", "uq"]:
        agg_df.loc[(agg_df["mode"] == mode) & (agg_df["engine"] == "pybncore"),
                   "speedup_smile_over_pybncore"] = speedups[mode]

    agg_csv = out_dir / "summary.csv"
    agg_df.to_csv(agg_csv, index=False)

    plot_png = out_dir / "benchmark_plot.png"
    _plot_results(agg_df, plot_png)

    report_md = out_dir / "report.md"
    with report_md.open("w") as f:
        f.write("# Smile vs pybncore Unified Benchmark\n\n")
        f.write(f"- Network: `{args.layers}x{args.width}`, max in-degree `{args.degree}`\n")
        f.write(f"- Target node: `{target_var_name}`\n")
        f.write(f"- Repeats: `{args.repeats}`\n")
        f.write(f"- Point rows: `{args.point_rows}`\n")
        f.write(f"- Batch rows: `{args.batch_rows}`\n")
        f.write(f"- UQ samples: `{args.uq_samples}`\n\n")
        f.write("## Median Speedups (Smile time / pybncore time)\n\n")
        for mode in ["point", "batch", "uq"]:
            f.write(f"- {mode}: `{speedups[mode]:.3f}x`\n")
        f.write("\n## Artifacts\n\n")
        f.write(f"- Raw runs: `{raw_csv.name}`\n")
        f.write(f"- Summary: `{agg_csv.name}`\n")
        f.write(f"- Plot: `{plot_png.name}`\n")

    print("Unified benchmark completed.")
    print(f"Output directory: {out_dir}")
    print(f"Raw runs: {raw_csv}")
    print(f"Summary: {agg_csv}")
    print(f"Plot: {plot_png}")
    print(f"Report: {report_md}")
    print("\nMedian times (seconds):")
    for mode in ["point", "batch", "uq"]:
        py = float(agg_df[(agg_df["mode"] == mode) & (agg_df["engine"] == "pybncore")]["median_seconds"])
        sm = float(agg_df[(agg_df["mode"] == mode) & (agg_df["engine"] == "smile")]["median_seconds"])
        print(f"  {mode}: pybncore={py:.6f}, smile={sm:.6f}, speedup(smile/pybncore)={speedups[mode]:.3f}x")


if __name__ == "__main__":
    main()
