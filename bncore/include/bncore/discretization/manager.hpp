#pragma once
// ============================================================================
//  DiscretizationManager — Neil–Tailor–Marquez dynamic discretization state.
//
//  Owns the bin grid and analytic CPD for every continuous variable.
//  Provides the four operations of the NTM outer loop:
//    (1) rebuild_cpts   — fill each continuous var's discrete CPT by
//                         integrating its CpdIntegrator over current bins
//    (2) compute_errors — per-bin entropy error from the posterior
//    (3) refine         — split top-error bins + merge low-error bins
//    (4) converged      — convergence check on max error / posterior KL
// ============================================================================
#include "bncore/discretization/cpd_integrator.hpp"
#include "bncore/graph/graph.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bncore {

struct ContinuousMetadata {
  NodeId id = 0;
  std::string name;
  // Support bounds: large finite clamps approximate ±infinity.
  double domain_lo = 0.0;
  double domain_hi = 0.0;
  // Bin edges: size = m+1.  Bin j covers [edges[j], edges[j+1]).
  std::vector<double> edges;
  // Ordered list of parents — duplicated here because the manager needs
  // to walk parent configurations when rebuilding CPTs.  Each entry is
  // either a continuous variable (registered in this manager) or a
  // discrete one (get states from Graph).
  std::vector<NodeId> parents;
  bool log_spaced = false;
  // Rare-event mode (Zhu-Collette 2015): reweights per-bin entropy error by
  // 1/max(p_j, p_floor), amplifying low-mass tail bins.  Also enables tail
  // preservation (outermost bins never removed) and threshold-aligned edges.
  bool rare_event_mode = false;
  // Query thresholds: edges are seeded here on the initial grid and after
  // every refinement so tail-mass integrals have a clean boundary.
  std::vector<double> thresholds;
  std::unique_ptr<CpdIntegrator> cpd;
  // Most recent posterior bin masses (length m) — updated by HybridEngine
  // between iterations, used by compute_errors.
  std::vector<double> posterior;
  // Previous-iteration posterior (length m) — used for KL convergence.
  // NOTE: may differ in length from posterior after refinement; we
  // interpolate onto the current grid when comparing.
  std::vector<double> prev_posterior;
  std::vector<double> prev_edges;

  // Move-only (unique_ptr members).
  ContinuousMetadata() = default;
  ContinuousMetadata(const ContinuousMetadata &) = delete;
  ContinuousMetadata &operator=(const ContinuousMetadata &) = delete;
  ContinuousMetadata(ContinuousMetadata &&) = default;
  ContinuousMetadata &operator=(ContinuousMetadata &&) = default;
};

class DiscretizationManager {
public:
  explicit DiscretizationManager(std::size_t max_bins_per_var = 40);

  // Non-copyable (owns unique_ptr<CpdIntegrator> through ContinuousMetadata).
  DiscretizationManager(const DiscretizationManager &) = delete;
  DiscretizationManager &operator=(const DiscretizationManager &) = delete;
  DiscretizationManager(DiscretizationManager &&) = default;
  DiscretizationManager &operator=(DiscretizationManager &&) = default;

  // ─── Registration ─────────────────────────────────────────────────────────
  // Register a continuous variable.  The variable itself is NOT added to the
  // graph here; the caller must pre-create it via graph.add_variable() with
  // placeholder states named "b0", "b1", ..., or call initialize_graph()
  // below which creates the placeholders matching the current bin grid.
  //
  // `parents` must match the parents previously added to the graph.
  // `initial_bins` is the starting bin count (will be refined).
  // `log_spaced` picks log-uniform edges in [domain_lo, domain_hi) for
  // positive-support variables (exponential, lognormal).
  void register_variable(NodeId var_id, const std::string &name,
                         std::vector<NodeId> parents,
                         std::unique_ptr<CpdIntegrator> cpd,
                         double domain_lo, double domain_hi,
                         std::size_t initial_bins = 8,
                         bool log_spaced = false,
                         bool rare_event_mode = false);

  // Declare a threshold of interest for `var_id` (e.g. a rare-event cut-off).
  // The initial grid is seeded with an edge exactly at `threshold`, and the
  // edge is re-inserted after every refinement if a split removes it.  Use
  // for queries of the form P(X < threshold).
  void add_threshold(NodeId var_id, double threshold);

  // Create/replace the graph-level states for every registered continuous
  // variable to match the current bin grid.  Call after all registrations
  // and BEFORE the first rebuild_cpts().
  void initialize_graph(Graph &graph) const;

  // ─── Outer-loop operations ────────────────────────────────────────────────

  // For each continuous variable, integrate its CpdIntegrator over every
  // (parent-config, child-bin) cell to produce a fresh CPT.  For each
  // parent-config, the resulting row is renormalised to sum to 1.
  void rebuild_cpts(Graph &graph);

  // Given bin-mass posteriors for some/all continuous variables, store
  // them and return the maximum per-bin entropy error across stored vars.
  // `posteriors` keys on NodeId; unregistered ids are ignored.
  double compute_errors(
      const std::unordered_map<NodeId, std::vector<double>> &posteriors);

  // Split top-K bins per variable (K = max(1, m/10)) at midpoint of the
  // highest-error bins, merge adjacent bins whose combined error is below
  // 0.1 × mean.  Respect max_bins_per_var_.  Also calls Graph::split_state()
  // as needed so the graph and CPT stay consistent.  Returns the number of
  // splits + merges applied across all variables.
  std::size_t refine(Graph &graph);

  // True if max_error < eps OR KL(prev, cur) < eps_kl on every registered
  // variable.  Must be called AFTER compute_errors() on the current iter.
  bool converged(double eps_entropy, double eps_kl) const;

  // ─── Accessors ────────────────────────────────────────────────────────────
  const std::vector<ContinuousMetadata> &variables() const { return vars_; }
  std::size_t max_bins_per_var() const { return max_bins_per_var_; }
  double last_max_error() const { return last_max_error_; }

  // Legacy API (kept for backward compatibility with existing example test).
  // These now also route through the CPT resize in Graph::split_state.
  bool should_split(const Graph &graph, NodeId var) const;
  void split_bin(Graph &graph, NodeId var, std::size_t state_idx);

private:
  std::size_t max_bins_per_var_;
  std::vector<ContinuousMetadata> vars_;
  // Cached lookup by NodeId → index in vars_.
  std::unordered_map<NodeId, std::size_t> index_of_;
  double last_max_error_ = 0.0;
};

} // namespace bncore
