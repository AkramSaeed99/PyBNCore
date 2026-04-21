---
name: hybrid-continuous
description: GUI patterns for PyBNCore's hybrid/continuous features — continuous node authoring, thresholds, dynamic discretization diagnostics, and continuous-posterior visualization. Use whenever working on hybrid-related dialogs, panels, or services.
---

# When to use
- Adding/editing a continuous node (normal, lognormal, uniform, exponential).
- Adding a deterministic continuous node (equation over parents).
- Adding a threshold (discretization boundary) on a continuous parent.
- Running `hybrid_query` and displaying `ContinuousPosterior` results.
- Showing DD (dynamic discretization) convergence diagnostics.

# Wrapper surface (recap)
- Distributions: `add_normal`, `add_lognormal`, `add_uniform`, `add_exponential`
- Deterministic: `add_deterministic(name, parents, expression, ...)`
- Threshold: `add_threshold(name, threshold)`
- Evidence: `set_continuous_evidence`, `clear_continuous_evidence`, `set_continuous_likelihood`
- Query: `wrapper.hybrid_query(nodes, ...)` → dict of `ContinuousPosterior` for continuous members
- `ContinuousPosterior` API: `mean`, `variance`, `std`, `support`, `cdf(x)`, `prob_less_than`, `prob_greater_than`, `prob_between(a, b)`, `quantile(q)`, `median`, `pdf(x)`, `plot(ax)`

# UI: add-continuous-node dialog
Tabs by distribution kind:
- **Normal** — μ, σ fields; preview of pdf on the right.
- **Lognormal** — μ, σ of underlying normal.
- **Uniform** — a, b.
- **Exponential** — λ.
- **Deterministic** — parent picker + expression editor (monospace, syntax-validated).
- **Threshold** — parent picker + threshold value + optional label.

All tabs show:
- Node name (validated against existing node IDs).
- Parents list (continuous-only for deterministic; any for threshold).
- Initial discretization: bin count (default 20) + strategy selector (uniform / quantile / adaptive).
- Live pdf/cdf preview via `pyqtgraph.PlotWidget`.
- Validate button runs service-side checks before enabling OK.

# UI: continuous evidence editor
- Each continuous node gets a row: numeric field (observed value) + "Clear" button.
- Soft/likelihood mode: curve editor that emits a discretized likelihood vector (delegates to `set_continuous_likelihood`).

# UI: continuous-posterior panel
For each queried continuous node, show:
- Header: name, mean ± std, support, mode bin.
- `pyqtgraph` plot with toggles: PDF, CDF, both.
- Interactive cursors:
  - x-cursor reads `cdf(x)` and `pdf(x)`.
  - Shaded band reads `prob_between(a, b)`.
- Quantile strip: inputs q ∈ {0.05, 0.25, 0.5, 0.75, 0.95} with computed x values (live-updates when the user changes q).
- Tail probability input: `P(X > x)` and `P(X < x)` panel.

# UI: DD convergence diagnostics
When `hybrid_query` iterates via dynamic discretization, surface per-iteration:
- Iteration count
- KL divergence between consecutive posteriors
- Current bin count per continuous node
- Convergence criterion met (yes/no)

Expose this by having `HybridService.run_hybrid_query` accept a callback and forward progress via the worker's `progress` signal.

# Threshold seeding / rare-event mode
- Advanced dialog that accepts a list of threshold values to seed the discretization grid.
- Include a "suggest from prior" button that samples `pdf(x)` via the current wrapper state and picks quantile-aligned thresholds.

# Service responsibilities (`services/hybrid_service.py`)
- Takes primitives + DTOs; never exposes raw `ContinuousPosterior` to views.
- Wraps `ContinuousPosterior` in `ContinuousPosteriorDTO` but keeps the raw object stored in the DTO for on-demand queries (CDF sampling, quantile lookups) without re-running inference.
- CDF grid: sample 256 evenly spaced points over `support` at DTO construction time for fast plotting; fall back to the raw object for exact queries.

# Worker rules
- `hybrid_query` iterates; always run via `HybridQueryWorker`. Never on the UI thread.
- Provide cancel — the worker checks a flag between DD iterations.

# Bans
- Don't store `ContinuousPosterior` objects in view code; they travel only in DTOs returned by the service.
- Don't recompute the CDF grid in the paint loop — use the cached grid in the DTO.
- Don't allow thresholds on purely discrete nodes — validate at the service layer before calling `add_threshold`.
