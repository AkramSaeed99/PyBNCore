---
name: bn-cpp-architecture-planner
description: Create concrete architecture and implementation plans for a C++ Bayesian network solver with exact inference on discretized models, dynamic discretization, batched execution, parallelization, and Python bindings.
---

# BN C++ Architecture Planner

## Purpose

Use this skill when the user wants to design, scope, or plan a C++ Bayesian network library, especially when the project includes:
- exact inference on discrete or discretized Bayesian networks
- dynamic discretization of continuous variables
- batched or vectorized query execution
- multithreading or SIMD-aware design
- Python interoperability for research workflows
- replacement of an existing BN library such as SMILE

This skill is for technical planning and architecture design. It is not for writing a vague research essay.

## Default assumptions

Unless the user states otherwise, assume the following:
- The implementation language is modern C++.
- The library is a compiled core with Python bindings.
- The solver must be callable from Python for notebooks, scripts, and workflows.
- The primary Python binding choice is nanobind.
- The build system is CMake.
- The target is a research-grade library that may later become a reusable package.
- Exact means exact inference on the current discretized network, not exact inference on the original continuous model.
- Repeated queries on one compiled model matter more than one-off queries.
- Sparse to moderate graph structure is the expected use case.
- Performance and memory locality are important design drivers.

## What to produce

When this skill is used, produce a direct and implementation-oriented plan with these sections:

1. Key assumptions
2. Executive recommendation
3. Proposed repository structure
4. Core modules and responsibilities
5. Internal data structures and memory layout
6. Exact inference engine design
7. Dynamic discretization design
8. Batched and vectorized execution design
9. Parallelization strategy
10. Python binding and package strategy
11. Build and tooling plan
12. Testing and benchmarking plan
13. MVP milestones
14. Risks, limitations, and non-goals

Do not skip sections unless the user explicitly narrows the request.

## Planning rules

### 1. Separate the system into layers

Default to these layers:
- core graph and metadata layer
- factor and evidence layer
- compilation and inference layer
- dynamic discretization layer
- batch execution layer
- Python binding layer

Keep these layers loosely coupled. Do not mix Python concerns into the inference core.

### 2. Prefer a small, buildable architecture

Prefer the smallest architecture that supports:
- exact inference on a compiled discrete model
- repeated evidence updates
- batch-aware factor operations
- future refinement for dynamic discretization

Do not propose an oversized framework.

### 3. Treat Python interoperability as a first-class requirement

Always include:
- binding strategy
- NumPy-oriented data exchange
- package layout
- wheel build direction
- examples of Python-facing API

Do not treat Python as an afterthought.

### 4. Be precise about exactness

Always state this clearly:
- inference is exact on the discretized model
- discretization introduces modeling approximation relative to the original continuous problem

Do not blur this distinction.

### 5. Be explicit about performance design

Always address:
- memory layout
- dense vs sparse factor storage
- immutable compiled structures where appropriate
- thread-local scratch state
- batch axis representation
- reuse of compiled inference state across repeated queries

### 6. Distinguish vectorization from parallelization

Always separate:
- vectorization over batch dimension
- SIMD inside kernels
- multithreaded parallel work across batch chunks or independent tasks
- graph-level synchronization constraints

Do not use these terms loosely.

### 7. Keep v1 realistic

For v1, prefer:
- discrete core first
- one primary exact inference engine
- a practical dynamic discretization loop
- minimal but solid Python bindings
- correctness and repeated-query performance over feature breadth

Postpone broad hybrid BN support, exotic factor types, and multiple inference backends unless the user asks.

## Required technical judgments

When making recommendations, explicitly decide and justify:
- junction tree vs variable elimination for v1
- nanobind vs pybind11 for bindings
- dense vs sparse vs mixed factor strategy
- how compiled structures are cached and reused
- what gets batch-awareness from the start
- what is deferred until later milestones

## Output style

Write like an engineer preparing a build plan:
- short sections
- direct statements
- concrete module names
- concrete class names where helpful
- milestone-oriented
- honest about risks

Avoid:
- buzzwords
- textbook filler
- generic introductions to Bayesian networks
- long motivational paragraphs

## Preferred repository skeleton

Use a structure close to this when proposing a repo layout, unless the user gives an existing repo:

- `include/`
- `src/`
- `python/`
- `tests/`
- `benchmarks/`
- `examples/`
- `docs/`
- `cmake/`

Inside the core, favor subdirectories such as:
- `graph/`
- `factors/`
- `inference/`
- `discretization/`
- `batch/`
- `util/`

## Preferred API style

Prefer an API with:
- explicit model construction
- explicit compile step
- reusable compiled solver object
- separate evidence objects
- batch query methods
- minimal hidden global state

Python-facing examples should look simple and research-friendly.

## Risk checklist

Always include at least these risks:
- clique explosion and treewidth growth
- discretization refinement cost
- inconsistent or unstable bin refinement criteria
- copying overhead across Python and C++
- lock contention or poor scaling in threaded execution
- memory blow-up from naive batch handling

## Non-goals checklist

Unless the user asks otherwise, classify these as later work:
- approximate inference engines
- learning BN structure from data
- full probabilistic programming features
- GPU offload
- every possible continuous distribution family
- distributed execution across multiple machines

## Final instruction

When using this skill, do not stop at high-level advice. Produce a concrete technical plan that an engineer could start implementing.