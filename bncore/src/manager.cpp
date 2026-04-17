#include "bncore/discretization/manager.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace bncore {

namespace {

// Build initial linearly- or log-spaced bin edges.
std::vector<double> make_initial_edges(double lo, double hi,
                                       std::size_t nbins, bool log_spaced) {
  std::vector<double> edges(nbins + 1);
  if (log_spaced) {
    if (lo <= 0.0) lo = 1e-12;
    if (hi <= lo) hi = lo * 10.0;
    const double log_lo = std::log(lo);
    const double log_hi = std::log(hi);
    const double step = (log_hi - log_lo) / static_cast<double>(nbins);
    for (std::size_t i = 0; i <= nbins; ++i)
      edges[i] = std::exp(log_lo + step * static_cast<double>(i));
  } else {
    const double step = (hi - lo) / static_cast<double>(nbins);
    for (std::size_t i = 0; i <= nbins; ++i)
      edges[i] = lo + step * static_cast<double>(i);
  }
  return edges;
}

// Midpoint of a bin (linear or log).
double bin_midpoint(double lo, double hi, bool log_spaced) {
  if (log_spaced && lo > 0.0 && hi > 0.0) {
    return std::exp(0.5 * (std::log(lo) + std::log(hi)));
  }
  return 0.5 * (lo + hi);
}

// New edge when splitting a bin (midpoint; log-midpoint if log-spaced).
double split_point(double lo, double hi, bool log_spaced) {
  return bin_midpoint(lo, hi, log_spaced);
}

// Placeholder state name for bin index i.
std::string bin_state_name(std::size_t i) {
  return "b" + std::to_string(i);
}

} // anonymous

DiscretizationManager::DiscretizationManager(std::size_t max_bins_per_var)
    : max_bins_per_var_(max_bins_per_var) {}

void DiscretizationManager::register_variable(
    NodeId var_id, const std::string &name, std::vector<NodeId> parents,
    std::unique_ptr<CpdIntegrator> cpd, double domain_lo, double domain_hi,
    std::size_t initial_bins, bool log_spaced, bool rare_event_mode) {
  if (initial_bins < 2)
    throw std::invalid_argument(
        "register_variable: initial_bins must be >= 2.");
  if (domain_hi <= domain_lo)
    throw std::invalid_argument(
        "register_variable: domain_hi must be > domain_lo.");
  if (!cpd)
    throw std::invalid_argument("register_variable: cpd must be non-null.");

  ContinuousMetadata meta;
  meta.id = var_id;
  meta.name = name;
  meta.parents = std::move(parents);
  meta.domain_lo = domain_lo;
  meta.domain_hi = domain_hi;
  meta.log_spaced = log_spaced;
  meta.rare_event_mode = rare_event_mode;
  meta.cpd = std::move(cpd);
  meta.edges = make_initial_edges(domain_lo, domain_hi, initial_bins,
                                   log_spaced);
  meta.posterior.assign(initial_bins, 0.0);

  index_of_[var_id] = vars_.size();
  vars_.push_back(std::move(meta));
}

void DiscretizationManager::add_threshold(NodeId var_id, double threshold) {
  auto it = index_of_.find(var_id);
  if (it == index_of_.end())
    throw std::invalid_argument(
        "add_threshold: variable not registered in manager.");
  auto &v = vars_[it->second];
  if (threshold <= v.domain_lo || threshold >= v.domain_hi)
    throw std::invalid_argument(
        "add_threshold: threshold must be strictly inside (domain_lo, "
        "domain_hi).");
  // Dedup + insert sorted.
  if (std::find(v.thresholds.begin(), v.thresholds.end(), threshold) !=
      v.thresholds.end())
    return;
  v.thresholds.push_back(threshold);
  std::sort(v.thresholds.begin(), v.thresholds.end());

  // Seed the current grid with this threshold as an edge.  If the closest
  // existing edge is within ~1e-9 (relative), skip; otherwise split the
  // containing bin at exactly the threshold.
  for (std::size_t j = 0; j + 1 < v.edges.size(); ++j) {
    const double lo = v.edges[j];
    const double hi = v.edges[j + 1];
    if (threshold > lo && threshold < hi) {
      const double width = hi - lo;
      if ((threshold - lo) / width < 1e-9 || (hi - threshold) / width < 1e-9)
        return;  // threshold already effectively at a boundary
      v.edges.insert(v.edges.begin() + j + 1, threshold);
      // Duplicate the bin's posterior mass across the two new bins.
      if (j < v.posterior.size()) {
        const double total = v.posterior[j];
        const double frac = (threshold - lo) / width;
        v.posterior[j] = total * frac;
        v.posterior.insert(v.posterior.begin() + j + 1, total * (1.0 - frac));
      }
      return;
    }
  }
}

void DiscretizationManager::initialize_graph(Graph &graph) const {
  for (const auto &v : vars_) {
    const std::size_t cur = graph.get_variable(v.id).states.size();
    const std::size_t nbins = v.edges.size() - 1;
    if (cur == nbins) continue;
    if (cur > nbins) {
      throw std::invalid_argument(
          "initialize_graph: variable '" + v.name + "' has " +
          std::to_string(cur) + " states in the graph but only " +
          std::to_string(nbins) + " bins registered.  Recreate the graph "
          "variable from scratch.");
    }
    // Grow graph state list to match the manager.  Repeated splits at
    // index 0 add states; the CPT will be overwritten by rebuild_cpts.
    std::size_t need = nbins - cur;
    for (std::size_t i = 0; i < need; ++i) {
      graph.split_state(v.id, 0, bin_state_name(0), bin_state_name(1));
    }
  }
}

// ----------------------------------------------------------------------------
// rebuild_cpts — re-integrate every continuous variable's CPT from its
// CpdIntegrator, given the current bin grid and parent bin midpoints.
// ----------------------------------------------------------------------------
// Helper: assign y ∈ [domain_lo, domain_hi) to the containing bin of `edges`.
// Returns the bin index, or clamps to the first/last bin if out of range.
static std::size_t bin_of(const std::vector<double> &edges, double y) {
  if (edges.size() <= 2) return 0;
  if (y <= edges.front()) return 0;
  if (y >= edges.back()) return edges.size() - 2;
  // Binary search — edges is sorted.
  std::size_t lo = 0, hi = edges.size() - 1;
  while (hi - lo > 1) {
    std::size_t mid = (lo + hi) >> 1;
    if (y < edges[mid]) hi = mid; else lo = mid;
  }
  return lo;
}

// Fill one CPT row for a deterministic node Y = g(parents) by Monte-Carlo
// sampling over the parent hyper-rectangle.  For each of `n_samples`
// random points, evaluate g, find the child bin, accumulate 1/n_samples.
static void fill_deterministic_row_mc(
    const ContinuousMetadata &v, const DeterministicCpd &det,
    const std::vector<std::size_t> &coord,
    const std::vector<bool> &pa_is_continuous,
    const std::vector<const ContinuousMetadata *> &pa_meta,
    double *row, std::size_t nbins_X) {
  const std::size_t n_samples = det.n_samples();
  // Simple LCG for reproducibility; seed by parent-config hash.
  std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
  for (std::size_t c : coord) rng = rng * 6364136223846793005ULL + c + 1;
  auto next_u = [&]() -> double {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    // Upper 24 bits → [0, 1)
    return static_cast<double>((rng >> 40) & 0xFFFFFF) / 16777216.0;
  };

  for (std::size_t b = 0; b < nbins_X; ++b) row[b] = 0.0;
  const double inv_n = 1.0 / static_cast<double>(n_samples);
  for (std::size_t s = 0; s < n_samples; ++s) {
    ParentBins pb;
    pb.continuous_values.reserve(coord.size());
    pb.discrete_states.reserve(coord.size());
    for (std::size_t p = 0; p < coord.size(); ++p) {
      if (pa_is_continuous[p]) {
        const auto &pm = *pa_meta[p];
        const std::size_t si = coord[p];
        const double lo = pm.edges[si];
        const double hi = pm.edges[si + 1];
        const double u = next_u();
        pb.continuous_values.push_back(lo + u * (hi - lo));
        pb.discrete_states.push_back(-1);
      } else {
        pb.continuous_values.push_back(0.0);
        pb.discrete_states.push_back(static_cast<int>(coord[p]));
      }
    }
    const double y = det.fn()(pb);
    if (!std::isfinite(y)) continue;
    const std::size_t bin = bin_of(v.edges, y);
    row[bin] += inv_n;
  }
}

// Fill one CPT row for a monotone deterministic node Y = g(parents) by
// interval arithmetic.  Evaluate g at the extreme corners of the parent
// hyper-rect to get [y_lo, y_hi], then distribute mass across Y-bins by
// overlap length.  Exact for monotone g.
static void fill_deterministic_row_monotone(
    const ContinuousMetadata &v, const DeterministicCpd &det,
    const std::vector<std::size_t> &coord,
    const std::vector<bool> &pa_is_continuous,
    const std::vector<const ContinuousMetadata *> &pa_meta,
    double *row, std::size_t nbins_X) {
  // Enumerate 2^k corners of the continuous-parent hyper-rect; evaluate g.
  std::vector<std::size_t> cont_idx;
  for (std::size_t p = 0; p < coord.size(); ++p)
    if (pa_is_continuous[p]) cont_idx.push_back(p);
  const std::size_t k = cont_idx.size();
  const std::size_t ncorners = (k == 0) ? 1 : (std::size_t{1} << k);

  double y_min = std::numeric_limits<double>::infinity();
  double y_max = -std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < ncorners; ++c) {
    ParentBins pb;
    pb.continuous_values.reserve(coord.size());
    pb.discrete_states.reserve(coord.size());
    for (std::size_t p = 0; p < coord.size(); ++p) {
      if (pa_is_continuous[p]) {
        const auto &pm = *pa_meta[p];
        const std::size_t si = coord[p];
        const std::size_t bit_pos =
            std::distance(cont_idx.begin(),
                          std::find(cont_idx.begin(), cont_idx.end(), p));
        const bool upper = (c >> bit_pos) & 1;
        pb.continuous_values.push_back(upper ? pm.edges[si + 1]
                                              : pm.edges[si]);
        pb.discrete_states.push_back(-1);
      } else {
        pb.continuous_values.push_back(0.0);
        pb.discrete_states.push_back(static_cast<int>(coord[p]));
      }
    }
    const double y = det.fn()(pb);
    if (std::isfinite(y)) {
      if (y < y_min) y_min = y;
      if (y > y_max) y_max = y;
    }
  }
  for (std::size_t b = 0; b < nbins_X; ++b) row[b] = 0.0;
  if (!std::isfinite(y_min) || !std::isfinite(y_max)) {
    const double u = 1.0 / static_cast<double>(nbins_X);
    for (std::size_t b = 0; b < nbins_X; ++b) row[b] = u;
    return;
  }
  const double span = y_max - y_min;
  if (span <= 0.0) {
    // Degenerate: g is constant over the cell — place all mass in one bin.
    const std::size_t bin = bin_of(v.edges, y_min);
    row[bin] = 1.0;
    return;
  }
  for (std::size_t b = 0; b < nbins_X; ++b) {
    const double lo = v.edges[b];
    const double hi = v.edges[b + 1];
    const double ov_lo = std::max(lo, y_min);
    const double ov_hi = std::min(hi, y_max);
    if (ov_hi > ov_lo) row[b] = (ov_hi - ov_lo) / span;
  }
}

void DiscretizationManager::rebuild_cpts(Graph &graph) {
  for (const auto &v : vars_) {
    const std::size_t nbins_X = v.edges.size() - 1;
    const std::size_t npar = v.parents.size();

    std::vector<std::size_t> pa_card(npar);
    std::vector<bool> pa_is_continuous(npar, false);
    std::vector<const ContinuousMetadata *> pa_meta(npar, nullptr);
    for (std::size_t p = 0; p < npar; ++p) {
      pa_card[p] = graph.get_variable(v.parents[p]).states.size();
      auto it = index_of_.find(v.parents[p]);
      if (it != index_of_.end()) {
        pa_is_continuous[p] = true;
        pa_meta[p] = &vars_[it->second];
      }
    }

    std::size_t n_configs = 1;
    for (std::size_t p = 0; p < npar; ++p) n_configs *= pa_card[p];

    std::vector<double> cpt(n_configs * nbins_X, 0.0);

    // Detect deterministic nodes — handled by a different code path.
    const auto *det = dynamic_cast<const DeterministicCpd *>(v.cpd.get());

    for (std::size_t cfg = 0; cfg < n_configs; ++cfg) {
      std::vector<std::size_t> coord(npar);
      {
        std::size_t rem = cfg;
        for (int p = static_cast<int>(npar) - 1; p >= 0; --p) {
          coord[p] = rem % pa_card[p];
          rem /= pa_card[p];
        }
      }

      double *row = cpt.data() + cfg * nbins_X;

      if (det) {
        if (det->monotone()) {
          fill_deterministic_row_monotone(v, *det, coord, pa_is_continuous,
                                          pa_meta, row, nbins_X);
        } else {
          fill_deterministic_row_mc(v, *det, coord, pa_is_continuous,
                                    pa_meta, row, nbins_X);
        }
        // Deterministic rows already sum to ~1 (or 0 if degenerate).
        double s = 0.0;
        for (std::size_t b = 0; b < nbins_X; ++b) s += row[b];
        if (s > 0.0) {
          for (std::size_t b = 0; b < nbins_X; ++b) row[b] /= s;
        } else {
          const double u = 1.0 / static_cast<double>(nbins_X);
          for (std::size_t b = 0; b < nbins_X; ++b) row[b] = u;
        }
        continue;
      }

      // Regular stochastic CPD path.
      ParentBins pb;
      pb.continuous_values.reserve(npar);
      pb.discrete_states.reserve(npar);
      for (std::size_t p = 0; p < npar; ++p) {
        if (pa_is_continuous[p]) {
          const auto &pm = *pa_meta[p];
          const std::size_t s = coord[p];
          const double mid = bin_midpoint(pm.edges[s], pm.edges[s + 1],
                                          pm.log_spaced);
          pb.continuous_values.push_back(mid);
          pb.discrete_states.push_back(-1);
        } else {
          pb.continuous_values.push_back(0.0);
          pb.discrete_states.push_back(static_cast<int>(coord[p]));
        }
      }

      double row_sum = 0.0;
      for (std::size_t b = 0; b < nbins_X; ++b) {
        double p = v.cpd->integrate(v.edges[b], v.edges[b + 1], pb);
        if (p < 0.0 || !std::isfinite(p)) p = 0.0;
        row[b] = p;
        row_sum += p;
      }
      if (row_sum > 0.0) {
        for (std::size_t b = 0; b < nbins_X; ++b) row[b] /= row_sum;
      } else {
        const double u = 1.0 / static_cast<double>(nbins_X);
        for (std::size_t b = 0; b < nbins_X; ++b) row[b] = u;
      }
    }

    graph.set_cpt(v.id, cpt);
  }
}

// ----------------------------------------------------------------------------
// compute_errors — per-bin entropy error (Kozlov/NTM 1-D form).
// ----------------------------------------------------------------------------
double DiscretizationManager::compute_errors(
    const std::unordered_map<NodeId, std::vector<double>> &posteriors) {
  double global_max = 0.0;
  for (auto &v : vars_) {
    auto it = posteriors.find(v.id);
    if (it == posteriors.end()) continue;
    const auto &p = it->second;
    if (p.size() + 1 != v.edges.size()) {
      throw std::invalid_argument(
          "compute_errors: posterior length for var '" + v.name +
          "' does not match bin count.");
    }
    v.prev_posterior = v.posterior;
    v.prev_edges = v.edges;
    v.posterior = p;

    const std::size_t m = p.size();
    std::vector<double> density(m);
    for (std::size_t j = 0; j < m; ++j) {
      const double w = v.edges[j + 1] - v.edges[j];
      density[j] = w > 0.0 ? p[j] / w : 0.0;
    }

    const double eps = 1e-12;
    const double p_floor = 1e-6;
    for (std::size_t j = 0; j < m; ++j) {
      const double w = v.edges[j + 1] - v.edges[j];
      const double f_mean = density[j];
      if (w <= 0.0 || f_mean <= 0.0) continue;
      const double f_lo = (j > 0) ? density[j - 1] : density[j];
      const double f_hi = (j + 1 < m) ? density[j + 1] : density[j];
      const double f_min = std::max(0.0, std::min({f_lo, f_mean, f_hi}));
      const double f_max = std::max({f_lo, f_mean, f_hi});
      const double term_hi = f_max *
          std::log((f_max + eps) / (f_mean + eps));
      const double term_lo = f_min *
          std::log((f_min + eps) / (f_mean + eps));
      double err = w * std::abs(term_hi - term_lo);
      // Zhu-Collette 2015: amplify rare-event bins so low-mass tails refine
      // before high-mass peaks.  Floor avoids division collapse.
      if (v.rare_event_mode) {
        err /= std::max(p[j], p_floor);
      }
      if (err > global_max) global_max = err;
    }
  }
  last_max_error_ = global_max;
  return global_max;
}

// ----------------------------------------------------------------------------
// refine — split the top-K highest-error bins per variable.
// ----------------------------------------------------------------------------
std::size_t DiscretizationManager::refine(Graph &graph) {
  std::size_t total = 0;
  for (auto &v : vars_) {
    const std::size_t m = v.edges.size() - 1;
    if (m >= max_bins_per_var_) continue;

    std::vector<double> density(m);
    for (std::size_t j = 0; j < m; ++j) {
      const double w = v.edges[j + 1] - v.edges[j];
      density[j] = (w > 0.0 && j < v.posterior.size()) ? v.posterior[j] / w
                                                       : 0.0;
    }
    std::vector<std::pair<double, std::size_t>> ranked;
    ranked.reserve(m);
    const double eps = 1e-12;
    const double p_floor = 1e-6;
    for (std::size_t j = 0; j < m; ++j) {
      const double w = v.edges[j + 1] - v.edges[j];
      const double f_mean = density[j];
      if (w <= 0.0 || f_mean <= 0.0) continue;
      const double f_lo = (j > 0) ? density[j - 1] : density[j];
      const double f_hi = (j + 1 < m) ? density[j + 1] : density[j];
      const double f_min = std::max(0.0, std::min({f_lo, f_mean, f_hi}));
      const double f_max = std::max({f_lo, f_mean, f_hi});
      const double term_hi = f_max * std::log((f_max + eps) / (f_mean + eps));
      const double term_lo = f_min * std::log((f_min + eps) / (f_mean + eps));
      double err = w * std::abs(term_hi - term_lo);
      if (v.rare_event_mode && j < v.posterior.size()) {
        err /= std::max(v.posterior[j], p_floor);
      }
      ranked.push_back({err, j});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](auto a, auto b) { return a.first > b.first; });

    const std::size_t budget = std::min<std::size_t>(
        max_bins_per_var_ - m, std::max<std::size_t>(1, m / 10));

    std::vector<std::size_t> targets;
    targets.reserve(budget);
    for (std::size_t r = 0; r < ranked.size() && targets.size() < budget; ++r)
      targets.push_back(ranked[r].second);
    // Sort descending so earlier splits don't shift later indices.
    std::sort(targets.begin(), targets.end(), std::greater<std::size_t>{});

    for (std::size_t bin : targets) {
      if (bin >= v.edges.size() - 1) continue;
      // Choose split point: if a declared threshold falls strictly inside
      // this bin, split exactly there (threshold-aligned seeding).
      // Otherwise split at the bin's midpoint (log or linear).
      const double lo = v.edges[bin];
      const double hi = v.edges[bin + 1];
      double sp = split_point(lo, hi, v.log_spaced);
      if (v.rare_event_mode) {
        for (double t : v.thresholds) {
          if (t > lo && t < hi) {
            const double frac = (t - lo) / (hi - lo);
            if (frac > 1e-6 && frac < 1.0 - 1e-6) {
              sp = t;
              break;
            }
          }
        }
      }
      v.edges.insert(v.edges.begin() + bin + 1, sp);
      if (bin < v.posterior.size()) {
        const double frac = (sp - lo) / (hi - lo);
        const double total_p = v.posterior[bin];
        v.posterior[bin] = total_p * frac;
        v.posterior.insert(v.posterior.begin() + bin + 1,
                            total_p * (1.0 - frac));
      }
      // Push split through the graph — resizes X's own CPT + all children's.
      graph.split_state(v.id, bin, bin_state_name(bin),
                        bin_state_name(bin + 1));
      ++total;
    }

    // Re-insert any threshold whose bin just got split away from the edge.
    // This keeps the tail-mass boundary clean across iterations.
    if (v.rare_event_mode) {
      for (double t : v.thresholds) {
        bool present = false;
        for (double e : v.edges) {
          if (std::abs(e - t) < 1e-12) { present = true; break; }
        }
        if (present) continue;
        // Find containing bin and re-split at threshold.
        for (std::size_t j = 0; j + 1 < v.edges.size(); ++j) {
          if (t > v.edges[j] && t < v.edges[j + 1]) {
            if (v.edges.size() - 1 >= max_bins_per_var_) break;
            const double lo = v.edges[j];
            const double hi = v.edges[j + 1];
            v.edges.insert(v.edges.begin() + j + 1, t);
            if (j < v.posterior.size()) {
              const double frac = (t - lo) / (hi - lo);
              const double total_p = v.posterior[j];
              v.posterior[j] = total_p * frac;
              v.posterior.insert(v.posterior.begin() + j + 1,
                                  total_p * (1.0 - frac));
            }
            graph.split_state(v.id, j, bin_state_name(j),
                              bin_state_name(j + 1));
            ++total;
            break;
          }
        }
      }
    }
  }
  return total;
}

// ----------------------------------------------------------------------------
// converged
// ----------------------------------------------------------------------------
bool DiscretizationManager::converged(double eps_entropy,
                                       double eps_kl) const {
  if (last_max_error_ < eps_entropy) return true;
  bool any_kl_ok = false;
  for (const auto &v : vars_) {
    if (v.prev_posterior.empty() || v.prev_edges != v.edges) return false;
    const std::size_t m = v.posterior.size();
    if (v.prev_posterior.size() != m) return false;
    double kl = 0.0;
    const double eps = 1e-12;
    for (std::size_t j = 0; j < m; ++j) {
      if (v.posterior[j] <= 0.0) continue;
      kl += v.posterior[j] *
            std::log((v.posterior[j] + eps) / (v.prev_posterior[j] + eps));
    }
    if (kl > eps_kl) return false;
    any_kl_ok = true;
  }
  return any_kl_ok;
}

// ─── Legacy API ──────────────────────────────────────────────────────────────
bool DiscretizationManager::should_split(const Graph &graph,
                                          NodeId var) const {
  return graph.get_variable(var).states.size() < max_bins_per_var_;
}

void DiscretizationManager::split_bin(Graph &graph, NodeId var,
                                       std::size_t state_idx) {
  if (!should_split(graph, var)) return;
  const auto &cur_states = graph.get_variable(var).states;
  if (state_idx >= cur_states.size()) return;
  const std::string old_name = cur_states[state_idx];
  graph.split_state(var, state_idx, old_name + "_low", old_name + "_high");
}

} // namespace bncore
