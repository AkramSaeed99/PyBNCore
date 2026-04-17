#include "bncore/inference/hybrid_engine.hpp"
#include "bncore/inference/compiler.hpp"
#include "bncore/inference/engine.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace bncore {

HybridEngine::HybridEngine(Graph &graph, DiscretizationManager &dm,
                           std::size_t num_threads)
    : graph_(graph), dm_(dm), num_threads_(num_threads) {}

// ─── Continuous evidence management ────────────────────────────────────────

void HybridEngine::set_evidence_continuous(NodeId var, double value) {
  // Find the registered continuous variable to validate domain.
  const ContinuousMetadata *meta = nullptr;
  for (const auto &v : dm_.variables()) {
    if (v.id == var) { meta = &v; break; }
  }
  if (!meta)
    throw std::invalid_argument(
        "set_evidence_continuous: variable not registered in the "
        "DiscretizationManager");
  if (value < meta->domain_lo || value >= meta->domain_hi)
    throw std::invalid_argument(
        "set_evidence_continuous: value outside [domain_lo, domain_hi)");
  pending_evidence_[var] = value;
  pending_soft_evidence_.erase(var);  // mutually exclusive
}

void HybridEngine::set_soft_evidence_continuous(
    NodeId var, std::function<double(double)> lik) {
  bool found = false;
  for (const auto &v : dm_.variables()) {
    if (v.id == var) { found = true; break; }
  }
  if (!found)
    throw std::invalid_argument(
        "set_soft_evidence_continuous: variable not registered in the "
        "DiscretizationManager");
  pending_soft_evidence_[var] = std::move(lik);
  pending_evidence_.erase(var);
}

void HybridEngine::clear_evidence_continuous(NodeId var) {
  pending_evidence_.erase(var);
  pending_soft_evidence_.erase(var);
}

// ─── Main DD loop ───────────────────────────────────────────────────────────

HybridEngine::RunResult HybridEngine::run(const int *evidence_row,
                                           std::size_t num_vars,
                                           const NodeId *query_vars,
                                           std::size_t num_queries,
                                           const RunConfig &cfg) {
  if (num_vars != graph_.num_variables())
    throw std::invalid_argument(
        "HybridEngine::run: num_vars must equal graph.num_variables()");

  // Prerequisite: graph state counts already match the bin counts.
  dm_.initialize_graph(graph_);

  // Build the query set = (user queries ∪ continuous variables).
  std::vector<NodeId> all_queries;
  all_queries.reserve(num_queries + dm_.variables().size());
  std::vector<char> seen(graph_.num_variables(), 0);
  for (std::size_t i = 0; i < num_queries; ++i) {
    NodeId q = query_vars[i];
    if (q < seen.size() && !seen[q]) {
      all_queries.push_back(q);
      seen[q] = 1;
    }
  }
  for (const auto &v : dm_.variables()) {
    if (v.id < seen.size() && !seen[v.id]) {
      all_queries.push_back(v.id);
      seen[v.id] = 1;
    }
  }
  if (all_queries.empty())
    throw std::invalid_argument(
        "HybridEngine::run: no query variables supplied.");

  RunResult result;
  std::unique_ptr<JunctionTree> jt;
  std::unique_ptr<BatchExecutionEngine> engine;

  // Base evidence row: caller's row OR all-unobserved.
  std::vector<int> base_ev(num_vars, -1);
  if (evidence_row) {
    for (std::size_t i = 0; i < num_vars; ++i) base_ev[i] = evidence_row[i];
  }

  for (std::size_t iter = 0; iter < cfg.max_iters; ++iter) {
    // 1. Rebuild continuous CPTs from the current bin grid.
    dm_.rebuild_cpts(graph_);

    // 2. Recompile JT and rebuild engine (state counts may have changed).
    jt = JunctionTreeCompiler::compile(graph_, "min_fill");
    engine = std::make_unique<BatchExecutionEngine>(*jt, num_threads_, 1);

    // 3. Translate value-based continuous evidence to bin indices on the
    // current grid, and install soft evidence vectors.
    std::vector<int> ev = base_ev;
    for (const auto &[var, value] : pending_evidence_) {
      const ContinuousMetadata *meta = nullptr;
      for (const auto &v : dm_.variables()) {
        if (v.id == var) { meta = &v; break; }
      }
      if (!meta) continue;
      // Binary search for the bin.
      const auto &edges = meta->edges;
      std::size_t j = 0;
      // Linear scan — bin count is small.
      for (std::size_t k = 0; k + 1 < edges.size(); ++k) {
        if (value >= edges[k] && value < edges[k + 1]) { j = k; break; }
        if (k + 2 == edges.size() && value >= edges[k + 1])
          j = k;  // value at domain_hi clamps to last bin
      }
      if (var < ev.size()) ev[var] = static_cast<int>(j);
    }
    // Soft evidence: build per-variable likelihood vector (one entry per bin).
    // Midpoint-rule integration of lik(x) over each bin.
    for (const auto &[var, lik] : pending_soft_evidence_) {
      const ContinuousMetadata *meta = nullptr;
      for (const auto &v : dm_.variables()) {
        if (v.id == var) { meta = &v; break; }
      }
      if (!meta) continue;
      const std::size_t m = meta->edges.size() - 1;
      std::vector<double> likelihoods(m, 0.0);
      for (std::size_t j = 0; j < m; ++j) {
        const double lo = meta->edges[j];
        const double hi = meta->edges[j + 1];
        const double mid = 0.5 * (lo + hi);
        double w = lik(mid);
        if (w < 0.0) w = 0.0;
        likelihoods[j] = w;
      }
      // Normalize for numerical stability (soft evidence is scale-invariant).
      double s = 0.0;
      for (double w : likelihoods) s += w;
      if (s > 0.0) {
        for (double &w : likelihoods) w /= s;
      }
      engine->set_soft_evidence(var, likelihoods.data(), m);
    }

    // 4. Build query offsets.
    std::vector<std::size_t> qvars_sz(all_queries.begin(), all_queries.end());
    std::vector<std::size_t> offsets;
    offsets.reserve(all_queries.size() + 1);
    offsets.push_back(0);
    for (NodeId q : all_queries)
      offsets.push_back(offsets.back() +
                        graph_.get_variable(q).states.size());
    const std::size_t total_states = offsets.back();

    std::vector<double> out(total_states, 0.0);
    engine->evaluate_multi(ev.data(), /*batch_size=*/1, num_vars,
                           qvars_sz.data(), qvars_sz.size(),
                           offsets.data(), out.data());

    // 5. Extract posteriors.
    std::unordered_map<NodeId, std::vector<double>> cont_posteriors;
    for (std::size_t qi = 0; qi < all_queries.size(); ++qi) {
      NodeId q = all_queries[qi];
      std::size_t a = offsets[qi];
      std::size_t b = offsets[qi + 1];
      std::vector<double> vec(out.begin() + a, out.begin() + b);
      result.posteriors[q] = vec;
      for (const auto &cv : dm_.variables()) {
        if (cv.id == q) {
          cont_posteriors[q] = vec;
          break;
        }
      }
    }

    // Snapshot edges for interpretation.
    result.edges.clear();
    for (const auto &cv : dm_.variables())
      result.edges[cv.id] = cv.edges;

    // 6. Entropy error + convergence.
    const double max_err = dm_.compute_errors(cont_posteriors);
    result.final_max_error = max_err;
    result.iterations_used = iter + 1;

    if (dm_.converged(cfg.eps_entropy, cfg.eps_kl)) break;
    if (iter + 1 == cfg.max_iters) break;

    // 7. Refine.
    dm_.refine(graph_);
  }

  return result;
}

} // namespace bncore
