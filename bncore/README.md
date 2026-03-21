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
For execution reviews, checkout `examples/test_api.py`.
```bash
# To test compilation and binding directly
source .venv/bin/activate
pip install -e .
python examples/test_api.py
```
