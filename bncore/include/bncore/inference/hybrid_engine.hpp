#pragma once
// ============================================================================
//  HybridEngine — dynamic discretization outer loop (Neil/Tailor/Marquez).
//
//  Wraps the existing discrete inference engine with an iterative bin
//  refinement loop.  Each iteration:
//    1. DiscretizationManager::rebuild_cpts()   — analytic CPD integration
//    2. Compile JT + run discrete inference     — existing machinery
//    3. Extract continuous posteriors           — per-variable bin masses
//    4. DiscretizationManager::compute_errors() — entropy-error bound
//    5. Check convergence; if not, refine and repeat.
// ============================================================================
#include "bncore/discretization/manager.hpp"
#include "bncore/graph/graph.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bncore {

class HybridEngine {
public:
  HybridEngine(Graph &graph, DiscretizationManager &dm,
               std::size_t num_threads = 1);

  struct RunConfig {
    std::size_t max_iters = 10;
    double eps_entropy = 1e-4;
    double eps_kl = 1e-4;
  };

  struct RunResult {
    std::size_t iterations_used = 0;
    double final_max_error = 0.0;
    // Per continuous variable: bin masses (length m).  Keyed by NodeId.
    std::unordered_map<NodeId, std::vector<double>> posteriors;
    // Current bin edges per continuous variable (for interpretation).
    std::unordered_map<NodeId, std::vector<double>> edges;
  };

  // Run DD inference.
  //   evidence_row: length = graph.num_variables(); -1 means unobserved.
  //                 For continuous variables the int is a bin index (after
  //                 the caller has mapped the observed value into a bin).
  //                 Pass nullptr for no evidence.  Combined with any
  //                 continuous evidence set via set_evidence_continuous().
  //   query_vars:   variables whose posteriors to report (continuous or
  //                 discrete).  Continuous vars always get their posterior
  //                 filled; query_vars is the explicit set the user cares
  //                 about for convergence and reporting.
  RunResult run(const int *evidence_row, std::size_t num_vars,
                const NodeId *query_vars, std::size_t num_queries,
                const RunConfig &cfg);

  // ─── Continuous evidence by VALUE (user-friendly) ───────────────────────
  // Hard evidence: pin `var` to the bin containing `value`.  Because bin
  // indices change across iterations due to refinement, the value is
  // re-resolved at each iteration.  Throws if `value` is outside the
  // variable's domain.
  void set_evidence_continuous(NodeId var, double value);

  // Soft evidence as a likelihood density λ(x) over the variable's domain.
  // At each iteration, ∫_bin_j λ(x) dx is computed (midpoint rule) and
  // passed to BatchExecutionEngine::set_soft_evidence() as the per-bin
  // likelihood vector.
  void set_soft_evidence_continuous(NodeId var,
                                     std::function<double(double)> lik);

  // Clear all value-based evidence and soft-evidence-by-likelihood for
  // one variable.
  void clear_evidence_continuous(NodeId var);

private:
  Graph &graph_;
  DiscretizationManager &dm_;
  std::size_t num_threads_;

  // Deferred continuous evidence, translated to bin indices per iteration.
  std::unordered_map<NodeId, double> pending_evidence_;
  std::unordered_map<NodeId, std::function<double(double)>>
      pending_soft_evidence_;
};

} // namespace bncore
