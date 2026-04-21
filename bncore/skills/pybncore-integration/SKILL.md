---
name: pybncore-integration
description: Wrap `PyBNCoreWrapper` behind service-layer classes and typed DTOs. Use whenever writing code in `services/` or exposing a new backend capability to the GUI.
---

# When to use
- Adding a new service method.
- Adding a DTO for a new result type.
- Handling evidence, compilation, inference, or analysis.

# Actual wrapper surface (from `pybncore/wrapper.py`)
Authoring / structure
- `PyBNCoreWrapper(model_path=None)`, `PyBNCoreWrapper.from_xdsl(path)`
- `wrapper.load(path)`
- `wrapper.nodes` → `List[str]`
- `wrapper.get_outcomes(node)`, `wrapper.parents(node)`, `wrapper.children(node)`
- `wrapper.add_noisy_max(node, states, parents, ...)`
- `wrapper.set_equation(node, expression, parents)`
- `wrapper.get_cpt_shaped(node)`, `wrapper.set_cpt(node, shaped, validate=False)`, `wrapper.set_cpt_batched(...)`

Evidence
- `wrapper.set_evidence(dict)`, `wrapper.clear_evidence()`
- `wrapper.set_soft_evidence(node, likelihoods)`, `wrapper.set_soft_evidence_matrix(...)`, `wrapper.clear_soft_evidence()`
- `wrapper.make_evidence_matrix(...)` for batch

Inference
- `wrapper.update_beliefs()` → compiles JT lazily
- `wrapper.query_p(node, state)`
- `wrapper.batch_query_marginals(nodes, evidence_matrix=None)`
- `wrapper.batch_query_map(evidence_matrix=None)`
- `wrapper.query_map(...)`

Analysis
- `wrapper.sensitivity(...)`, `wrapper.sensitivity_ranking(...)`
- `wrapper.value_of_information(query_node, candidate_nodes=None)`

Hybrid / continuous
- `wrapper.add_normal / add_lognormal / add_uniform / add_exponential / add_deterministic / add_threshold`
- `wrapper.set_continuous_evidence / clear_continuous_evidence / set_continuous_likelihood`
- `wrapper.hybrid_query(nodes, ...)` → returns `ContinuousPosterior` instances for continuous vars

IO
- `from pybncore.io import read_xdsl`

# Rules
- UI code **must not** import `pybncore` directly; it imports service classes only.
- Every service method takes primitives/DTOs and returns a typed DTO. No `PyBNCoreWrapper` object ever crosses the service boundary.
- Catch `RuntimeError`, `ValueError`, and `KeyError` thrown by the wrapper; re-raise as domain exceptions (`CompileError`, `EvidenceError`, `QueryError`, `IOError`) that carry a user-facing message.

# DTO catalog (defined in `domain/results.py`)
```python
@dataclass(frozen=True)
class PosteriorResult:
    node: str
    states: tuple[str, ...]
    probabilities: tuple[float, ...]
    evidence_snapshot: Mapping[str, str]

@dataclass(frozen=True)
class BatchPosteriorResult:
    nodes: tuple[str, ...]
    matrix: np.ndarray          # shape (n_rows, sum_state_counts) or dict-of-arrays
    row_evidence: tuple[Mapping[str, str], ...]

@dataclass(frozen=True)
class MAPResult:
    assignment: Mapping[str, str]
    log_probability: float

@dataclass(frozen=True)
class SensitivityEntry:
    parameter: str
    sensitivity: float
    note: str = ""

@dataclass(frozen=True)
class VOIEntry:
    candidate: str
    voi: float

@dataclass(frozen=True)
class ContinuousPosteriorDTO:
    name: str
    mean: float
    std: float
    support: tuple[float, float]
    cdf_grid: tuple[tuple[float, float], ...]   # (x, F(x)) samples
    raw: ContinuousPosterior = field(repr=False) # keep original for on-demand queries
```

# Service layer template
```python
class InferenceService:
    def __init__(self, session: ModelSession) -> None:
        self._session = session   # owns the wrapper

    def compile(self) -> CompileStats:
        try:
            self._session.wrapper.update_beliefs()
        except RuntimeError as e:
            raise CompileError(str(e)) from e
        return CompileStats.from_wrapper(self._session.wrapper)

    def query_single(self, node: str, evidence: Mapping[str, str]) -> PosteriorResult:
        w = self._session.wrapper
        try:
            w.set_evidence(dict(evidence))
            w.update_beliefs()
            states = w.get_outcomes(node)
            probs = tuple(w.query_p(node, s) for s in states)
            return PosteriorResult(node, tuple(states), probs, dict(evidence))
        except (KeyError, ValueError) as e:
            raise QueryError(str(e)) from e
```

# Performance
- Cache compile state on `ModelSession`; only call `update_beliefs()` after a structural/parameter change.
- Use `batch_query_marginals` when a scenario table has > 1 row — never loop over rows in Python.
- Reuse evidence matrices; build via `wrapper.make_evidence_matrix(...)`.

# Validation rules (before any wrapper call)
- Evidence keys exist in `wrapper.nodes`; values are in `wrapper.get_outcomes(node)`.
- CPT shape matches `wrapper.get_cpt_shaped(node).shape` before `set_cpt`.
- Continuous evidence values are finite floats.
- Soft-evidence likelihoods sum to a finite positive number.

# Bans
- Don't swallow wrapper exceptions silently; always convert to a domain error.
- Don't expose the raw `ContinuousPosterior` object to views — wrap it in `ContinuousPosteriorDTO` (the DTO may keep it for on-demand `cdf/quantile` calls inside the service).
- Don't call `update_beliefs()` from views or viewmodels.
