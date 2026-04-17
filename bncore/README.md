# pybncore

A highly optimized, high-performance C++ Bayesian Network library explicitly designed for *vectorized, batched queries* and *exact exact probabilistic inference* over discretized domains.

This repository serves as a backend engine intended to solve thousands or millions of scenarios simultaneously (like risk modeling or monte carlo scenario evaluations) using a mathematical compiler that translates graphical networks into executable mathematical SIMD sequences.

## 🎯 Architecture Overview For Code Reviewers

This repository is built with a strictly separated compilation and execution phase strategy. It optimizes probabilistic inference by aggressively pushing allocations, node traversing, and tree logic to a pre-computation compilation phase, leaving a flat, chunked matrix pipeline for execution.

The general system architecture splits into four macro-domains:
1. **Graph Topological Modeling:** `Graph` structure, definitions, and `VariableMetadata`.
2. **JunctionTree Compilation:** Morphing the input Directed Acyclic Graph (DAG) over moralization and triangulation algorithms into an execution-friendly Clique Tree.
3. **Shafer-Shenoy Message Passings (Inference):** Extracting Marginal Probabilities over the compiled tree utilizing Collect and Distribute phases.
4. **Platform Python Bindings:** `nanobind` extensions allowing direct zero-copy access from python applications.

### 1. The Core Graph and Factors
Located in `include/bncore/graph` and `include/bncore/factors`.
- **`Graph`**: Handles Bayesian dependencies. Nodes store `VariableMetadata` (the dimensions and states of explicit probability matrices).
- **`DenseTensor` & `Factor`**: Represents multidimensional probability tables (CPTs or Potentials) mapped continuously in 1D memory.
    - Crucially, the **Innermost Dimension is reserved for the `batch_size`**. This implies adjacent memories represent identical scenarios across different batch states ensuring maximum CPU cache-locality when performing evaluations.

### 2. The Junction Tree Compiler
Located in `include/bncore/inference/compiler.hpp` and `include/bncore/inference/junction_tree.hpp`.
Exact inference directly on DAGs is computationally messy. To fix this, `JunctionTreeCompiler::compile` preconfigures the model:
- **Moralization**: Marries graph parents ensuring shared factor combinations.
- **Triangulation (Min-Fill heuristic)**: Converts graph to a chordal representation to define Maximum Spanning Trees.
- **Kruskal's Algorithm**: Extracts Maximal Cliques to formulate the `JunctionTree` - an immutable exact acyclic representation ready for inference.

### 3. Batched Execution Engine & Workspace
Located in `include/bncore/inference/engine.hpp` and `include/bncore/inference/workspace.hpp`.

The execution framework is what differentiates `pybncore` from typical educational Bayesian tools.

- **`BatchExecutionEngine`**: Handles incoming multi-scenario numpy queries (`nanobind::ndarray<int>`). It orchestrates chunking (e.g., executing matrices 1024-rows at a time) and leverages standard C++ `std::async` threads to parallelize batched workflows.
- **`BatchWorkspace`**: The thread-local execution memory. During a chunk operation, it handles the following memory loop:
    1. Loads the immutable `JunctionTree` base factors.
    2. Overlays the requested **hard evidence** from Python leveraging dynamically generated `indicator` factors.
    3. Runs the **Shafer-Shenoy message-passing algorithm**.
        * **Phase 1 (Collect)**: Leaves propagate factor messages iteratively resolving towards the Root clique.
        * **Phase 2 (Distribute)**: Root passes calibrated messages back outwards to all leaves ensuring global normalization accuracy.
- **`BumpAllocator`**: Zero-allocation Scratchpad.
    * Real-time Bayesian Message passings require allocating thousands of intermediate `Factor` tensors per second.
    * The `BumpAllocator` provides a contiguous chunk of pre-allocated system memory, shifting a single integer pointer to dispense memory blocks.
    * Once the execution chunk is successfully solved, the pointer resets to `0`, effectively erasing memory histories in $O(1)$ without a single OS interrupt (`malloc` or `free`).

### 4. SIMD Math Engine Acceleration
Look at `Factor::multiply` and `Factor::marginalize` inside `src/factor.cpp`.
Memory layout guarantees batch evaluations iterate strictly consecutively across dimensions: `a_ptr[b] * b_ptr[b]`. The `#pragma omp simd` compiler injections enforce actual silicon-level vector instructions ensuring math loops unroll automatically spanning 128-bit/256-bit registers directly across arrays.

### 5. Python Binding Integrations (Nanobind)
Located in `src_python/main.cpp`.
We expose explicit Py-Types. `nanobind` was adopted specifically for natively embedding and verifying C-contiguous numpy bindings without deep copy extractions.
Upon invoking `BatchExecutionEngine::evaluate` from Python, `nb::gil_scoped_release` drops the interpreter lock enabling instantaneous physical multithreading spanning available logical processors.

## 🛠️ Testing the Framework

```bash
# Build and install the extension
pip install -e .

# Run the full test suite
pytest tests/ -v

# Browse the example gallery
python examples/01_hello_discrete.py              # classic rain/sprinkler
python examples/02_sensitivity_and_voi.py         # sensitivity + VoI
python examples/03_hybrid_gaussian_chain.py       # Kalman-like chain
python examples/04_reliability_rare_event.py     # ⭐ PRA flagship example
python examples/05_diagnostic_with_soft_evidence.py
```

## 🚀 Advanced Inference Features

`pybncore` now natively supports advanced reasoning and sensitivity analytics:

```python
# 1. Soft / Virtual Evidence
# Scale the posterior probabilities directly without clamping to 1.0 or 0.0
wrapper.set_soft_evidence("Sensor", {"High": 0.8, "Medium": 0.15, "Low": 0.05})

# 2. Maximum A Posteriori (MAP) Inference
# Find the single most likely joint state configuration globally
best_state = wrapper.query_map({"Observation": "True"})

# 3. Parameter Sensitivity Analysis
# Evaluate how P(Target) responds to continuously sweeping a specific CPT parameter
sensitivity_results = wrapper.sensitivity(
    query_node="Patient_Status", query_state="Sick",
    target_node="Test_Accuracy", parent_config=("Gen_3_Scanner",), target_state="High",
    sweep_range=np.linspace(0.8, 1.0, 20)
)
# Rank all CPT parameters by their local derivative on the target query
rankings = wrapper.sensitivity_ranking(query_node="Patient_Status", query_state="Sick")

# 4. Value of Information (VoI)
# Rank observation candidates by Expected Entropy Reduction (Mutual Information)
voi_scores = wrapper.value_of_information("Patient_Status", candidate_nodes=["Blood_Test", "X_Ray"])
```

## 🧮 Hybrid Bayesian Networks (Continuous Variables)

For models with continuous variables — loads, temperatures, degradation
rates — `pybncore` implements the **Neil–Tailor–Marquez dynamic
discretization** algorithm, with **Zhu-Collette rare-event reweighting**
for reliability / PRA workloads.

```python
from pybncore import PyBNCoreWrapper

w = PyBNCoreWrapper()

# Continuous variables use type-specific registration
w.add_lognormal("R", log_mu=-2.0, log_sigma=0.5, domain=(1e-4, 10.0),
                rare_event_mode=True)
w.add_normal   ("L", mu=5.0, sigma=1.0, domain=(0.0, 10.0))
w.add_uniform  ("C", a=8.0, b=12.0, domain=(8.0, 12.0))

# Functional / deterministic nodes: Y = f(parents)
w.add_deterministic(
    "stress", parents=["R", "L", "C"],
    fn=lambda r, l, c: r * l / c,
    domain=(0.0, 5.0),
)

# Rare-event threshold — forces an exact bin edge at 0.8
w.add_threshold("stress", 0.8)

# Hard evidence by value (no manual bin-index arithmetic)
w.set_continuous_evidence({"L": 5.5})

# Run the DD outer loop and get a ContinuousPosterior back
result = w.hybrid_query(["stress"])
stress = result["stress"]

p_fail = stress.prob_greater_than(0.8)   # tail-accurate rare-event probability
mean   = stress.mean()
q95    = stress.quantile(0.95)
```

See [**docs/dd_tutorial.md**](docs/dd_tutorial.md) for the full DD
tutorial and [**examples/04_reliability_rare_event.py**](examples/04_reliability_rare_event.py)
for an end-to-end PRA example.


## 📜 License
This project is licensed under a **Proprietary License**. 
Copyright (c) 2026 **Akram Batikh** <asbatikh@ncsu.edu>. 
All Rights Reserved.
