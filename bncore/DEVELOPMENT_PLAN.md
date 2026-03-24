# PyBNCore — Development Roadmap

> Last updated: 2026-03-23

This document tracks planned features, their priority, and implementation status.  
It is a living document; update status fields and add notes as work progresses.

---

## Priority Tiers

| Symbol | Meaning |
|--------|---------|
| 🔴 **P1** | High priority — missing core inference capability |
| 🟠 **P2** | Medium priority — model expressiveness |
| 🟡 **P3** | Lower priority — learning and I/O |
| ✅ | Implemented |
| 🚧 | In progress |
| ⬜ | Planned |

---

## 🔴 P1 — Core Inference Capabilities

### 1. Soft / Virtual Evidence ⬜

**What:** Instead of hard-zeroing non-matching potential entries, soft evidence weights each state of a variable with a user-provided **likelihood vector** $\lambda = [\lambda_0, \ldots, \lambda_{k-1}]$.  
The potential entries are multiplied by $\lambda[\text{state}]$ rather than zeroed when state ≠ observed.

**When to use:** Sensor readings with known noise models, fused/uncertain observations, Bayesian data likelihoods.

**API target (C++):**
```cpp
// workspace.hpp
void set_soft_evidence(NodeId var, const double* likelihoods, std::size_t n_states);
void clear_soft_evidence();
```

**API target (Python):**
```python
wrapper.set_soft_evidence("SensorA", {"True": 0.9, "False": 0.1})
wrapper.set_soft_evidence_matrix("SensorA", likelihood_matrix)  # shape (B, k)
```

**Hard evidence is a special case:** $\lambda_s = 1$ if $s = \text{observed}$, else $0$.  
The implementation therefore replaces the current hard-evidence zeroing with a unified likelihood-multiply path.

**Status:** ⬜ Planned  
**Files changed:** `workspace.hpp`, `workspace.cpp`, `engine.hpp`, `engine.cpp`, `wrapper.py`

---

### 2. MAP / MPE Inference ⬜

**What:** Maximum A Posteriori (MAP) / Most Probable Explanation (MPE) finds the joint assignment of all (or a subset of) variables that has the highest posterior probability:
$$\hat{x} = \arg\max_x P(x \mid e)$$

This requires **max-product** message passing (replacing sum-product marginalization with max) during the collect pass, followed by a **traceback** pass to decode the argmax states.

**When to use:** Fault diagnosis (most likely fault given symptoms), root-cause analysis, optimal joint prediction.

**API target:**
```cpp
// engine.hpp
void evaluate_map(const int* evidence, std::size_t batch_size,
                  std::size_t num_vars, int* output_states);
```

```python
# Returns: Dict[str, str]  (node → most probable state)
states = wrapper.query_map(evidence={"SensorA": "True"})
# Returns: np.ndarray shape (B, N_nodes) for batched MAP
states = wrapper.batch_query_map(evidence_matrix)
```

**Status:** ⬜ Planned  
**Files changed:** `workspace.hpp`, `workspace.cpp`, `engine.hpp`, `engine.cpp`, `wrapper.py`

---

### 3. Sensitivity Analysis ⬜

**What:** Computes how much a target posterior $P(Q \mid e)$ changes in response to perturbations of a CPT parameter $\theta_{i,j}$.  Two modes:

- **One-way:** vary a single CPT entry; report $\Delta P(Q)$ as a function of $\Delta\theta$.
- **What-if sweep:** for a CPT row $[\theta_0, \ldots, \theta_{k-1}]$, find the range of each $\theta_i$ over which the MAP assignment or posterior ranking remains unchanged.

**API target:**
```python
result = wrapper.sensitivity(
    query_node="Outcome",
    query_state="True",
    target_node="Cause",
    target_parent_config={"Parent": "A"},
    sweep_range=(0.0, 1.0),
    n_points=50,
)
# result: {"theta": np.ndarray, "posterior": np.ndarray}
```

**Status:** ⬜ Planned  
**Files changed:** `wrapper.py` (pure Python post-calibration sweep), optionally `engine.cpp` for batched parameter sweeps.

---

### 4. Value of Information (VoI) ⬜

**What:** For a target query node $Q$ and a candidate observation node $V$ (not yet observed), computes the **Expected Value of Perfect Information** (EVPI):
$$\text{VoI}(V) = \mathbb{E}_V[H(Q \mid V)] - H(Q)$$
where $H(\cdot)$ is Shannon entropy (or another divergence measure).

VoI ranks which unobserved variable would most reduce uncertainty about the query.

**API target:**
```python
ranking = wrapper.value_of_information(
    query_node="Diagnosis",
    candidate_nodes=["Test1", "Test2", "Test3"],  # None = all unobserved
)
# Returns: List[Tuple[str, float]]  sorted by VoI descending
```

**Implementation note:** VoI is computed entirely by running repeated calibration sweeps in Python over the already-compiled JT. No C++ changes required unless batched VoI is needed for performance.

**Status:** ⬜ Planned  
**Files changed:** `wrapper.py` (new method, pure Python over existing engine)

---

## 🟠 P2 — Model Expressiveness

### 5. Noisy-OR / Noisy-MAX / DeMorgan Gates ⬜

**What:** Compact parameterization of large CPTs via independence-of-causal-influence (ICI) models. Reduces CPT size from exponential $O(k^{n_\text{parents}})$ to linear $O(n_\text{parents})$.

**Status:** ⬜ Planned  
**Files changed:** `graph.hpp`, `graph.cpp`, `workspace.cpp` (CPT expansion on compilation)

---

### 6. Influence Diagrams / Decision Networks ⬜

**What:** Adds decision nodes (actions) and utility nodes (payoffs) to the BN.  
Enables rational decision-making under uncertainty via variable elimination over the decision tree.

**Status:** ⬜ Planned  
**Files changed:** New `decision/` module, `compiler.cpp` (extension)

---

### 7. Equation / Functional Nodes ⬜

**What:** Child CPT defined by an analytic expression (e.g., $P(Y=y|x) = f(x)$) instead of a raw table. Critical for hybrid physical/probabilistic models.

**Status:** ⬜ Planned  
**Files changed:** `graph.hpp`, `graph.cpp`, new `defequation` support

---

## 🟡 P3 — Learning and I/O

### 8. Parameter Learning (EM) ⬜

Learn CPTs from data using the Expectation-Maximization algorithm.

**Status:** ⬜ Planned

---

### 9. Structure Learning ⬜

Learn the DAG from data (PC algorithm, hill-climbing, etc.).

**Status:** ⬜ Planned

---

### 10. XDSL / BIF File I/O ⬜

Read/write standard BN file formats for interoperability with GeNIe, Netica, pgmpy, and SMILE.  
XDSL read is partially implemented in `io.py`; BIF write and full XDSL round-trip are missing.

**Status:** ⬜ Planned  
**Files changed:** `io.py`

---

### 11. Approximate Inference ⬜

Loopy belief propagation or particle filtering for networks with high treewidth.  
This would be a **differentiator over SMILE** (which only does exact inference).

**Status:** ⬜ Planned

---

## Differentiators vs. SMILE

| Feature | SMILE | PyBNCore |
|---------|-------|---------|
| Exact inference (JT) | ✅ | ✅ |
| Batched / vectorized inference (parallel rows) | ❌ | ✅ |
| Zero-alloc hot path, lazy COW propagation | ❌ | ✅ |
| Native NumPy / Python-first API | ❌ | ✅ |
| Soft / virtual evidence | ✅ | ⬜ P1 |
| MAP / MPE inference | ✅ | ⬜ P1 |
| Sensitivity analysis | ✅ | ⬜ P1 |
| Value of Information | ✅ | ⬜ P1 |
| Noisy gates (OR/MAX/DeMorgan) | ✅ | ⬜ P2 |
| Influence diagrams | ✅ | ⬜ P2 |
| Parameter learning (EM) | ✅ | ⬜ P3 |
| Structure learning | ✅ | ⬜ P3 |
| Approximate inference (loopy BP) | ❌ | ⬜ P3 |
| Open-source / embeddable | ❌ commercial | ✅ |
