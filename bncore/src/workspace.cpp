#include "bncore/inference/workspace.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace bncore {

// ---------------------------------------------------------------------------
// Static helpers — map builders
// ---------------------------------------------------------------------------
static std::vector<std::size_t> make_strides(const std::vector<std::size_t> &shape) {
  std::vector<std::size_t> st(shape.size());
  if (!shape.empty()) {
    st.back() = 1;
    for (int i = (int)shape.size() - 2; i >= 0; --i)
      st[i] = st[i + 1] * shape[i + 1];
  }
  return st;
}

static std::vector<NodeId> combine_scopes(const std::vector<NodeId> &a,
                                          const std::vector<NodeId> &b,
                                          std::vector<int> &b_in_c) {
  std::vector<NodeId> c = a;
  b_in_c.resize(b.size(), -1);
  for (std::size_t j = 0; j < b.size(); ++j) {
    auto it = std::find(c.begin(), c.end(), b[j]);
    if (it == c.end()) { b_in_c[j] = (int)c.size(); c.push_back(b[j]); }
    else                  b_in_c[j] = (int)(it - c.begin());
  }
  return c;
}

// Build flat index map: product scope → source scope
// Returns product_states count.
static std::size_t build_pot_map_u32(const std::vector<NodeId>   &prod_scope,
                                     const std::vector<std::size_t> &prod_shape,
                                     const std::vector<NodeId>   &src_scope,
                                     const std::vector<std::size_t> &src_shape,
                                     std::vector<uint32_t>        &pot_map) {
  std::size_t prod_states = 1;
  for (auto s : prod_shape) prod_states *= s;
  auto prod_st = make_strides(prod_shape);
  auto src_st  = make_strides(src_shape);
  pot_map.resize(prod_states);

  for (std::size_t pi = 0; pi < prod_states; ++pi) {
    std::size_t si = 0, rem = pi;
    for (std::size_t d = 0; d < prod_scope.size(); ++d) {
      std::size_t coord = rem / prod_st[d];
      rem -= coord * prod_st[d];
      for (std::size_t sd = 0; sd < src_scope.size(); ++sd)
        if (src_scope[sd] == prod_scope[d]) { si += coord * src_st[sd]; break; }
    }
    pot_map[pi] = (uint32_t)si;
  }
  return prod_states;
}

// Build marginalisation map: src scope → dst scope (dst is a subset of src)
static void build_marg_map_u32(const std::vector<NodeId>   &src_scope,
                               const std::vector<std::size_t> &src_shape,
                               const std::vector<NodeId>   &dst_scope,
                               const std::vector<std::size_t> &dst_shape,
                               std::vector<uint32_t>        &marg_map) {
  std::size_t src_st_tot = 1;
  for (auto s : src_shape) src_st_tot *= s;
  auto src_st = make_strides(src_shape);
  auto dst_st = make_strides(dst_shape);
  marg_map.resize(src_st_tot);

  for (std::size_t si = 0; si < src_st_tot; ++si) {
    std::size_t di = 0, rem = si;
    for (std::size_t d = 0; d < src_scope.size(); ++d) {
      std::size_t coord = rem / src_st[d];
      rem -= coord * src_st[d];
      for (std::size_t dd = 0; dd < dst_scope.size(); ++dd)
        if (dst_scope[dd] == src_scope[d]) { di += coord * dst_st[dd]; break; }
    }
    marg_map[si] = (uint32_t)di;
  }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
BatchWorkspace::BatchWorkspace(const JunctionTree &jt, std::size_t batch_size)
    : jt_(jt), batch_size_(batch_size) {

  std::size_t n = jt_.cliques().size();
  root_clique_  = ~std::size_t(0);
  parent_of_.assign(n, ~std::size_t(0));
  children_of_.assign(n, {});

  // BFS to orient the tree
  std::vector<bool> visited(n, false);
  std::vector<std::size_t> stack;
  for (std::size_t start = 0; start < n; ++start) {
    if (visited[start]) continue;
    if (root_clique_ == ~std::size_t(0)) root_clique_ = start;
    stack.push_back(start);
    visited[start] = true;
    while (!stack.empty()) {
      std::size_t u = stack.back(); stack.pop_back();
      distribute_order_.push_back(u);
      for (std::size_t v : jt_.cliques()[u].neighbor_cliques)
        if (!visited[v]) {
          visited[v] = true;
          parent_of_[v] = u;
          children_of_[u].push_back(v);
          stack.push_back(v);
        }
    }
  }
  collect_order_ = distribute_order_;
  std::reverse(collect_order_.begin(), collect_order_.end());

  build_node_to_clique_map();
  build_message_schedule();
  rebuild_clique_potentials(batch_size_);
}

// ---------------------------------------------------------------------------
// build_node_to_clique_map
// ---------------------------------------------------------------------------
void BatchWorkspace::build_node_to_clique_map() {
  node_to_clique_.clear();
  node_in_cliques_.clear();
  for (std::size_t i = 0; i < jt_.cliques().size(); ++i)
    for (NodeId nd : jt_.cliques()[i].scope) {
      node_to_clique_.emplace(nd, i);  // first occurrence = query clique
      node_in_cliques_[nd].push_back(i);
    }
}

// ---------------------------------------------------------------------------
// build_message_schedule — pre-compute all ops + query marg maps
// ---------------------------------------------------------------------------
void BatchWorkspace::build_message_schedule() {
  std::size_t n = jt_.cliques().size();

  // Clique scope shapes
  std::vector<std::vector<std::size_t>> cshapes(n);
  clique_states_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (NodeId nd : jt_.cliques()[i].scope)
      cshapes[i].push_back(jt_.graph()->get_variable(nd).states.size());
    clique_states_[i] = 1;
    for (auto s : cshapes[i]) clique_states_[i] *= s;
  }

  // Sepset sizes
  sepset_size_.assign(n, 1);
  auto sepset_scope_shape = [&](std::size_t u, std::size_t v)
      -> std::pair<std::vector<NodeId>, std::vector<std::size_t>> {
    const auto &us = jt_.cliques()[u].scope, &vs = jt_.cliques()[v].scope;
    std::vector<NodeId> sep;
    std::set_intersection(us.begin(), us.end(), vs.begin(), vs.end(), std::back_inserter(sep));
    std::vector<std::size_t> sh;
    for (NodeId nd : sep) sh.push_back(jt_.graph()->get_variable(nd).states.size());
    return {sep, sh};
  };

  for (std::size_t u = 0; u < n; ++u) {
    if (parent_of_[u] == ~std::size_t(0)) continue;
    auto [sep, sh] = sepset_scope_shape(u, parent_of_[u]);
    sepset_size_[u] = 1; for (auto s : sh) sepset_size_[u] *= s;
  }

  // Message buffers
  up_msg_buf_.resize(n);  down_msg_buf_.resize(n);
  up_msg_size_.resize(n, 0); down_msg_size_.resize(n, 0);
  base_up_buf_.resize(n); base_down_buf_.resize(n);

  // Calibrated potential buffers
  cal_pot_buf_.resize(n);  cal_pot_size_.resize(n);  base_cal_buf_.resize(n);

  // P3: evidence scratch (per-clique, lazy resize)
  ev_scratch_.resize(n);
  ev_scratch_active_.assign(n, false);

  // ── CollectOps ──────────────────────────────────────────────────────────
  collect_ops_.resize(n);
  for (std::size_t u : collect_order_) {
    if (parent_of_[u] == ~std::size_t(0)) continue;
    CollectOp &op = collect_ops_[u];
    op.child_cliques = children_of_[u];

    std::vector<NodeId> prod_scope = jt_.cliques()[u].scope;
    std::vector<std::size_t> prod_shape = cshapes[u];

    op.child_maps.resize(children_of_[u].size());
    for (std::size_t ci = 0; ci < children_of_[u].size(); ++ci) {
      auto [sep, sh] = sepset_scope_shape(children_of_[u][ci], u);
      std::vector<int> b_in_c;
      auto new_scope = combine_scopes(prod_scope, sep, b_in_c);
      std::vector<std::size_t> new_shape;
      for (auto nd : new_scope) {
        auto it = std::find(prod_scope.begin(), prod_scope.end(), nd);
        new_shape.push_back(it != prod_scope.end()
          ? prod_shape[it - prod_scope.begin()]
          : jt_.graph()->get_variable(nd).states.size());
      }
      prod_scope = new_scope; prod_shape = new_shape;
    }

    op.product_states = build_pot_map_u32(prod_scope, prod_shape,
                                          jt_.cliques()[u].scope, cshapes[u],
                                          op.pot_map);
    for (std::size_t ci = 0; ci < children_of_[u].size(); ++ci) {
      auto [sep, sh] = sepset_scope_shape(children_of_[u][ci], u);
      build_pot_map_u32(prod_scope, prod_shape, sep, sh, op.child_maps[ci]);
    }
    auto [sep_scope, sep_shape] = sepset_scope_shape(u, parent_of_[u]);
    build_marg_map_u32(prod_scope, prod_shape, sep_scope, sep_shape, op.marg_map);

    max_product_states_ = std::max(max_product_states_, op.product_states);
  }

  // ── DistributeOps ────────────────────────────────────────────────────────
  distribute_ops_.resize(n);
  for (std::size_t u : distribute_order_) {
    if (parent_of_[u] == ~std::size_t(0)) continue;
    DistributeOp &op = distribute_ops_[u];
    std::size_t p = parent_of_[u];
    op.parent_clique = p;

    std::vector<NodeId> prod_scope = jt_.cliques()[p].scope;
    std::vector<std::size_t> prod_shape = cshapes[p];
    op.has_parent_down = (parent_of_[p] != ~std::size_t(0));

    op.sibling_cliques.clear();
    for (std::size_t s : children_of_[p]) if (s != u) op.sibling_cliques.push_back(s);

    for (std::size_t s : op.sibling_cliques) {
      auto [sep, sh] = sepset_scope_shape(s, p);
      std::vector<int> b_in_c;
      auto new_scope = combine_scopes(prod_scope, sep, b_in_c);
      std::vector<std::size_t> new_shape;
      for (auto nd : new_scope) {
        auto it = std::find(prod_scope.begin(), prod_scope.end(), nd);
        new_shape.push_back(it != prod_scope.end()
          ? prod_shape[it - prod_scope.begin()]
          : jt_.graph()->get_variable(nd).states.size());
      }
      prod_scope = new_scope; prod_shape = new_shape;
    }

    op.product_states = build_pot_map_u32(prod_scope, prod_shape,
                                          jt_.cliques()[p].scope, cshapes[p],
                                          op.pot_map);
    if (op.has_parent_down) {
      auto [sep, sh] = sepset_scope_shape(p, parent_of_[p]);
      build_pot_map_u32(prod_scope, prod_shape, sep, sh, op.down_map);
    }
    op.sibling_maps.resize(op.sibling_cliques.size());
    for (std::size_t si = 0; si < op.sibling_cliques.size(); ++si) {
      auto [sep, sh] = sepset_scope_shape(op.sibling_cliques[si], p);
      build_pot_map_u32(prod_scope, prod_shape, sep, sh, op.sibling_maps[si]);
    }
    auto [sep_scope, sep_shape] = sepset_scope_shape(u, p);
    build_marg_map_u32(prod_scope, prod_shape, sep_scope, sep_shape, op.marg_map);

    max_product_states_ = std::max(max_product_states_, op.product_states);
  }

  // ── AssembleOps ──────────────────────────────────────────────────────────
  assemble_ops_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    AssembleOp &op = assemble_ops_[i];
    op.child_cliques = children_of_[i];
    op.has_down = (parent_of_[i] != ~std::size_t(0));
    op.product_states = clique_states_[i];

    if (op.has_down) {
      auto [sep, sh] = sepset_scope_shape(i, parent_of_[i]);
      build_pot_map_u32(jt_.cliques()[i].scope, cshapes[i], sep, sh, op.down_map);
    }
    op.child_maps.resize(children_of_[i].size());
    for (std::size_t ci = 0; ci < children_of_[i].size(); ++ci) {
      auto [sep, sh] = sepset_scope_shape(children_of_[i][ci], i);
      build_pot_map_u32(jt_.cliques()[i].scope, cshapes[i], sep, sh,
                        op.child_maps[ci]);
    }
  }

  // ── P1: Persistent scratch buffer sized to worst-case product ────────────
  max_product_states_ = std::max(max_product_states_, std::size_t(1));
  scratch_buf_.resize(max_product_states_, 0.0);

  // ── P2: Pre-baked per-clique per-variable marginalisation maps ────────────
  clique_marg_info_.resize(n);
  for (std::size_t ci = 0; ci < n; ++ci) {
    CliqueMargInfo &mi = clique_marg_info_[ci];
    const auto &scope = jt_.cliques()[ci].scope;
    std::size_t K = scope.size();
    mi.n_states_total = clique_states_[ci];
    mi.state_of_dim.assign(K, std::vector<uint8_t>(mi.n_states_total));

    auto strides = make_strides(cshapes[ci]);
    for (std::size_t flat = 0; flat < mi.n_states_total; ++flat) {
      std::size_t rem = flat;
      for (std::size_t d = 0; d < K; ++d) {
        uint8_t coord = (uint8_t)(rem / strides[d]);
        rem -= coord * strides[d];
        mi.state_of_dim[d][flat] = coord;
      }
    }
  }

  // ── Phase-A: size scratch flag vectors (pre-sized = zero per-call alloc) ──
  ev_clique_.assign(n, 0);
  up_changed_.assign(n, 0);
  down_changed_.assign(n, 0);
  cal_changed_.assign(n, 0);

  resize_runtime_buffers(batch_size_);
}

void BatchWorkspace::resize_runtime_buffers(std::size_t target_batch_size) {
  const std::size_t effective_batch_size = std::max<std::size_t>(1, target_batch_size);
  const std::size_t n = jt_.cliques().size();

  for (std::size_t u = 0; u < n; ++u) {
    if (parent_of_[u] == ~std::size_t(0)) {
      up_msg_buf_[u].clear();
      down_msg_buf_[u].clear();
      base_up_buf_[u].clear();
      base_down_buf_[u].clear();
      up_msg_size_[u] = 0;
      down_msg_size_[u] = 0;
      continue;
    }

    const std::size_t msg_size = sepset_size_[u] * effective_batch_size;
    up_msg_buf_[u].assign(msg_size, 0.0);
    down_msg_buf_[u].assign(msg_size, 0.0);
    base_up_buf_[u].assign(msg_size, 0.0);
    base_down_buf_[u].assign(msg_size, 0.0);
    up_msg_size_[u] = msg_size;
    down_msg_size_[u] = msg_size;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t pot_size = clique_states_[i] * effective_batch_size;
    cal_pot_buf_[i].assign(pot_size, 0.0);
    base_cal_buf_[i].assign(pot_size, 0.0);
    cal_pot_size_[i] = pot_size;
  }

  scratch_buf_.assign(max_product_states_ * effective_batch_size, 0.0);
}


// ---------------------------------------------------------------------------
// rebuild_clique_potentials
// ---------------------------------------------------------------------------
void BatchWorkspace::snapshot_base_potentials() {
  base_clique_potentials_.clear();
  base_clique_potentials_.reserve(clique_potentials_.size());
  for (const Factor &pot : clique_potentials_) {
    Factor owned(pot.scope(), pot.tensor().shape());
    std::copy(pot.tensor().data(), pot.tensor().data() + pot.tensor().size(),
              owned.tensor().data());
    base_clique_potentials_.push_back(std::move(owned));
  }
}

void BatchWorkspace::rebuild_clique_potentials(std::size_t target_batch_size) {
  if (target_batch_size == 0) {
    throw std::invalid_argument("BatchWorkspace batch_size must be > 0.");
  }

  resize_runtime_buffers(target_batch_size);

  // This is construction-time only — ok to use large temporary allocator.
  // We use a local temporary std::vector-backed buffer so this is safe for
  // any target_batch_size.
  clique_potentials_.clear();
  clique_potentials_.reserve(jt_.cliques().size());
  bool uses_batch_offset = false;

  for (const auto &clique : jt_.cliques()) {
    const Factor &unbatched = clique.base_potential;
    std::vector<std::size_t> new_sizes;
    for (std::size_t i = 0; i < unbatched.scope().size(); ++i)
      new_sizes.push_back(unbatched.tensor().shape()[i]);
    new_sizes.push_back(target_batch_size);

    Factor batched(unbatched.scope(), new_sizes);  // heap-owned
    batched.tensor().fill(1.0);

    for (NodeId n_id : clique.assigned_cpts) {
      std::vector<NodeId> family;
      for (NodeId p : jt_.graph()->get_parents(n_id)) family.push_back(p);
      family.push_back(n_id);
      std::vector<std::size_t> fam_sizes;
      for (NodeId p : family) fam_sizes.push_back(jt_.graph()->get_variable(p).states.size());
      std::size_t fam_states = 1; for (auto s : fam_sizes) fam_states *= s;

      std::vector<std::size_t> new_fam = fam_sizes;
      new_fam.push_back(target_batch_size);
      Factor cpt(family, new_fam);

      const auto &probs = jt_.graph()->get_variable(n_id).cpt;
      if (probs.empty()) {
        throw std::invalid_argument("CPT for variable '" +
                                    jt_.graph()->get_variable(n_id).name +
                                    "' is empty.");
      }
      if (probs.size() % fam_states != 0) {
        throw std::invalid_argument(
            "CPT geometry mismatch for variable '" +
            jt_.graph()->get_variable(n_id).name + "'. Expected scalar size " +
            std::to_string(fam_states) + " or batched size " +
            std::to_string(fam_states) + " * B, but got " +
            std::to_string(probs.size()) + ".");
      }
      const std::size_t cpt_batch_size = probs.size() / fam_states;
      if (cpt_batch_size == 0) {
        throw std::invalid_argument("CPT for variable '" +
                                    jt_.graph()->get_variable(n_id).name +
                                    "' has invalid batch size 0.");
      }
      if (cpt_batch_size != 1 &&
          !(cpt_batch_size == target_batch_size && cpt_batch_offset_ == 0) &&
          !(cpt_batch_size > target_batch_size &&
            cpt_batch_offset_ + target_batch_size <= cpt_batch_size)) {
        throw std::invalid_argument(
            "Batched CPT batch size mismatch for variable '" +
            jt_.graph()->get_variable(n_id).name + "': CPT batch size is " +
            std::to_string(cpt_batch_size) + ", workspace batch size is " +
            std::to_string(target_batch_size) + ", batch offset is " +
            std::to_string(cpt_batch_offset_) + ".");
      }

      double *d = cpt.tensor().data();
      if (cpt_batch_size == 1) {
        for (std::size_t s = 0; s < fam_states; ++s)
          for (std::size_t b = 0; b < target_batch_size; ++b)
            d[s * target_batch_size + b] = probs[s];
      } else if (cpt_batch_size == target_batch_size) {
        std::copy(probs.begin(), probs.end(), d);
      } else {
        uses_batch_offset = true;
        for (std::size_t s = 0; s < fam_states; ++s) {
          const double *src = probs.data() + s * cpt_batch_size + cpt_batch_offset_;
          double *dst = d + s * target_batch_size;
          std::copy(src, src + target_batch_size, dst);
        }
      }
      batched = batched.multiply(cpt);
    }
    Factor owned(unbatched.scope(), new_sizes);
    std::copy(batched.tensor().data(),
              batched.tensor().data() + batched.tensor().size(),
              owned.tensor().data());
    clique_potentials_.push_back(std::move(owned));
  }

  snapshot_base_potentials();
  base_batch_size_ = target_batch_size;
  base_batch_offset_ = cpt_batch_offset_;
  base_depends_on_offset_ = uses_batch_offset;
  is_first_calibration_ = true;
  clear_evidence();
  calibrate();
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
void BatchWorkspace::reset(std::size_t new_batch_size,
                           std::size_t cpt_batch_offset) {
  if (new_batch_size == 0) {
    throw std::invalid_argument("BatchWorkspace batch_size must be > 0.");
  }

  batch_size_ = new_batch_size;
  cpt_batch_offset_ = cpt_batch_offset;
  clear_evidence();

  // Restore evidence scratch tracking
  std::fill(ev_scratch_active_.begin(), ev_scratch_active_.end(), false);

  if (batch_size_ == base_batch_size_ && !base_clique_potentials_.empty() &&
      (!base_depends_on_offset_ || cpt_batch_offset_ == base_batch_offset_)) {
    clique_potentials_.clear();
    clique_potentials_.reserve(base_clique_potentials_.size());
    for (const Factor &bp : base_clique_potentials_)
      clique_potentials_.push_back(
          Factor(bp.scope(), bp.tensor().shape(),
                 const_cast<double*>(bp.tensor().data())));
    return;
  }
  rebuild_clique_potentials(batch_size_);
}

// ---------------------------------------------------------------------------
// Hard evidence (unchanged API)
// ---------------------------------------------------------------------------
void BatchWorkspace::set_evidence_matrix(const int *ev, std::size_t num_vars) {
  evidence_matrix_ = ev; evidence_num_vars_ = num_vars;
}
void BatchWorkspace::clear_evidence() {
  evidence_matrix_ = nullptr; evidence_num_vars_ = 0;
}

// ---------------------------------------------------------------------------
// Soft / virtual evidence
// ---------------------------------------------------------------------------
void BatchWorkspace::set_soft_evidence(NodeId var, const double *likelihoods,
                                       std::size_t n_states) {
  // Validate variable exists in graph.
  const std::size_t card = jt_.graph()->get_variable(var).states.size();
  if (n_states != card)
    throw std::invalid_argument(
        "set_soft_evidence: n_states (" + std::to_string(n_states) +
        ") does not match variable cardinality (" + std::to_string(card) + ").");
  for (std::size_t s = 0; s < n_states; ++s)
    if (likelihoods[s] < 0.0)
      throw std::invalid_argument(
          "set_soft_evidence: likelihood[" + std::to_string(s) +
          "] is negative (" + std::to_string(likelihoods[s]) + ").");
  soft_evidence_scalar_[var].assign(likelihoods, likelihoods + n_states);
  // Remove any batched soft evidence for this variable — scalar takes priority.
  soft_evidence_matrix_.erase(var);
}

void BatchWorkspace::set_soft_evidence_matrix(NodeId var,
                                               const double *likelihoods_matrix,
                                               std::size_t n_states) {
  const std::size_t card = jt_.graph()->get_variable(var).states.size();
  if (n_states != card)
    throw std::invalid_argument(
        "set_soft_evidence_matrix: n_states (" + std::to_string(n_states) +
        ") does not match variable cardinality (" + std::to_string(card) + ").");
  const std::size_t B = batch_size_;
  for (std::size_t i = 0; i < B * n_states; ++i)
    if (likelihoods_matrix[i] < 0.0)
      throw std::invalid_argument(
          "set_soft_evidence_matrix: negative likelihood at flat index " +
          std::to_string(i) + ".");
  soft_evidence_matrix_[var].assign(likelihoods_matrix,
                                    likelihoods_matrix + B * n_states);
  // Remove scalar soft evidence for this variable — matrix takes priority.
  soft_evidence_scalar_.erase(var);
}

void BatchWorkspace::clear_soft_evidence() {
  soft_evidence_scalar_.clear();
  soft_evidence_matrix_.clear();
}

// ---------------------------------------------------------------------------
// Barren node pruning — set_query_scope
//
// Marks which cliques are on the path from root to any clique containing a
// query variable.  Used by calibrate() B=1 path to skip distribute/assemble
// for barren subtrees that don't contribute to the queried marginals.
// ---------------------------------------------------------------------------
void BatchWorkspace::set_query_scope(const NodeId *vars, std::size_t num_vars) {
  const std::size_t n = jt_.cliques().size();

  if (query_relevant_.size() != n) {
    query_relevant_.resize(n, 0);
    is_query_clique_.resize(n, 0);
  }

  std::memset(query_relevant_.data(), 0, n);
  std::memset(is_query_clique_.data(), 0, n);

  if (num_vars == 0) {
    has_query_scope_ = false;
    return;
  }

  // Mark cliques that directly contain query variables.
  for (std::size_t i = 0; i < num_vars; ++i) {
    auto it = node_to_clique_.find(vars[i]);
    if (it != node_to_clique_.end()) {
      is_query_clique_[it->second] = 1;
    }
  }

  // Walk from each query clique to root, marking the path as relevant.
  // Early-exit when hitting an already-marked ancestor (its path to root
  // is already covered).
  for (std::size_t ci = 0; ci < n; ++ci) {
    if (!is_query_clique_[ci]) continue;
    std::size_t u = ci;
    while (u != ~std::size_t(0) && !query_relevant_[u]) {
      query_relevant_[u] = 1;
      u = parent_of_[u];
    }
  }

  has_query_scope_ = true;
}

// ---------------------------------------------------------------------------
// apply_soft_evidence_to_clique
//
// Applies soft evidence (or hard evidence converted to lambda vectors) to the
// potential of clique ci.  Returns true if any likelihood was applied.
//
// Design: both hard and soft evidence are handled by multiplying the potential
// entry at flat index f and batch row b by lambda[b][state_of_dim[d][f]].
// This is the unified path — hard evidence is a lambda=[0..1..0] special case.
//
// Key invariants maintained:
//   - ev_scratch_[ci] is used as a working copy (lazily sized).
//   - clique_potentials_[ci] is replaced with a Factor wrapping the scratch.
//   - ev_clique_[ci] is set to 1 when any change is made.
//   - If all entries sum to 0 (B==1), throws "Inconsistent evidence".
// ---------------------------------------------------------------------------
bool BatchWorkspace::apply_soft_evidence_to_clique(std::size_t ci) {
  const Factor  &pot   = clique_potentials_[ci];
  const auto    &scope = pot.scope();
  const auto    &shape = pot.tensor().shape();
  const std::size_t K  = scope.size();
  const std::size_t B  = batch_size_;

  // Collect which dimensions (variable positions in the clique scope) have
  // either hard or soft evidence.
  struct EvidenceDim {
    std::size_t pos;          // dimension in clique scope
    NodeId      nid;
    bool        is_scalar;   // true → same likelihood for all batch rows
  };
  std::vector<EvidenceDim> ev_dims;
  ev_dims.reserve(K);

  for (std::size_t d = 0; d < K; ++d) {
    NodeId nid = scope[d];
    bool has_hard = (evidence_matrix_ && nid < evidence_num_vars_);
    bool has_scalar_soft = (soft_evidence_scalar_.count(nid) > 0);
    bool has_matrix_soft = (soft_evidence_matrix_.count(nid) > 0);

    if (!has_hard && !has_scalar_soft && !has_matrix_soft) continue;

    // If we have both hard evidence and soft evidence for the same variable,
    // soft evidence takes precedence (it subsumes hard as a special case).
    if (has_scalar_soft || has_matrix_soft) {
      ev_dims.push_back({d, nid, has_scalar_soft});
    } else {
      // Hard evidence: check at least one batch row has a non-negative obs.
      bool any_obs = false;
      for (std::size_t b = 0; b < B && !any_obs; ++b)
        if (evidence_matrix_[b * evidence_num_vars_ + nid] >= 0) any_obs = true;
      if (any_obs) ev_dims.push_back({d, nid, /*is_scalar=*/true});
    }
  }
  if (ev_dims.empty()) return false;

  // Copy potential into scratch buffer.
  const std::size_t total = clique_states_[ci];
  std::vector<double> &scratch = ev_scratch_[ci];
  scratch.resize(total * B);
  const double *src = pot.tensor().data();
  std::copy(src, src + total * B, scratch.data());

  const CliqueMargInfo &mi = clique_marg_info_[ci];

  // Apply each evidence dimension.
  for (const EvidenceDim &ed : ev_dims) {
    const std::size_t d   = ed.pos;
    const NodeId      nid = ed.nid;
    const std::size_t card = shape[d];

    bool has_scalar_soft = soft_evidence_scalar_.count(nid) > 0;
    bool has_matrix_soft = soft_evidence_matrix_.count(nid) > 0;

    for (std::size_t flat = 0; flat < total; ++flat) {
      const std::size_t state = mi.state_of_dim[d][flat];
      double *dst = scratch.data() + flat * B;

      if (has_scalar_soft) {
        // Same lambda for all batch rows.
        const double *lam = soft_evidence_scalar_.at(nid).data();
        const double  factor = lam[state];
        if (factor == 1.0) continue;  // hot-path skip for hard evidence matching state
#pragma omp simd
        for (std::size_t b = 0; b < B; ++b) dst[b] *= factor;

      } else if (has_matrix_soft) {
        // Per-row lambda.  lam[b * card + state]
        const double *lam = soft_evidence_matrix_.at(nid).data();
        for (std::size_t b = 0; b < B; ++b) {
          dst[b] *= lam[b * card + state];
        }

      } else {
        // Hard evidence: convert to lambda on the fly.
        for (std::size_t b = 0; b < B; ++b) {
          const int obs = evidence_matrix_[b * evidence_num_vars_ + nid];
          if (obs < 0) continue;  // unobserved in this row
          if (static_cast<std::size_t>(obs) >= card)
            throw std::invalid_argument(
                "Hard evidence state index out of range for variable '" +
                jt_.graph()->get_variable(nid).name + "'.");
          if (static_cast<std::size_t>(obs) != state) dst[b] = 0.0;
        }
      }
    }
  }

  // Inconsistency check for the B==1 case — throw immediately.
  if (B == 1) {
    double sum = 0.0;
    for (std::size_t f = 0; f < total; ++f) sum += scratch[f];
    if (sum == 0.0)
      throw std::invalid_argument("Inconsistent evidence: zero-sum potential in clique.");
  }

  clique_potentials_[ci] = Factor(scope, shape, scratch.data());
  ev_scratch_active_[ci] = true;
  ev_clique_[ci]         = 1;
  return true;
}


// ---------------------------------------------------------------------------
// calibrate — ZERO heap allocation in the hot path
// ---------------------------------------------------------------------------
void BatchWorkspace::calibrate() {
  const std::size_t n = jt_.cliques().size();
  const std::size_t B = batch_size_;

  // Phase-A: clear workspace-owned flag arrays — memset, zero heap allocation.
  std::memset(ev_clique_.data(),    0, n);
  std::memset(up_changed_.data(),   0, n);
  std::memset(down_changed_.data(), 0, n);
  std::memset(cal_changed_.data(),  0, n);

  // ── Evidence application ─────────────────────────────────────────────────
  // Unified path: handles hard evidence, soft evidence, and combinations.
  // apply_soft_evidence_to_clique() sets ev_clique_[ci] and updates
  // clique_potentials_[ci] in place.  We call it for every clique (it returns
  // immediately if no evidence touches that clique).
  if (evidence_matrix_ || !soft_evidence_scalar_.empty() || !soft_evidence_matrix_.empty()) {
    for (std::size_t ci = 0; ci < n; ++ci) {
      if (ev_clique_[ci]) continue;  // already processed (guard for multi-pass)
      apply_soft_evidence_to_clique(ci);
    }
  }


  // ── Collect: leaves → root ───────────────────────────────────────────────
  double *scratch = scratch_buf_.data();
  if (B == 1) {
    for (std::size_t u : collect_order_) {
      if (parent_of_[u] == ~std::size_t(0)) continue;
      bool needs = ev_clique_[u];
      for (std::size_t c : children_of_[u]) if (up_changed_[c]) { needs = true; break; }
      if (!needs && !is_first_calibration_) {
        std::copy(base_up_buf_[u].begin(), base_up_buf_[u].end(), up_msg_buf_[u].begin());
        continue;
      }
      const CollectOp &op = collect_ops_[u];
      const double *pot = clique_potentials_[u].tensor().data();
      const std::size_t PS = op.product_states;
      for (std::size_t pi = 0; pi < PS; ++pi) scratch[pi] = pot[op.pot_map[pi]];
      for (std::size_t ci = 0; ci < op.child_cliques.size(); ++ci) {
        const double *cm = up_msg_buf_[op.child_cliques[ci]].data();
        const uint32_t *cmap = op.child_maps[ci].data();
        for (std::size_t pi = 0; pi < PS; ++pi) scratch[pi] *= cm[cmap[pi]];
      }
      double *umsg = up_msg_buf_[u].data();
      const std::size_t US = up_msg_size_[u];
      std::fill(umsg, umsg + US, 0.0);
      const uint32_t *mm = op.marg_map.data();
      for (std::size_t pi = 0; pi < PS; ++pi) umsg[mm[pi]] += scratch[pi];
      up_changed_[u] = 1;
    }

    // ── Distribute: root → leaves ──────────────────────────────────────────
    for (std::size_t u : distribute_order_) {
      if (parent_of_[u] == ~std::size_t(0)) {
        bool root_needs = ev_clique_[u];
        for (std::size_t c : children_of_[u]) if (up_changed_[c]) { root_needs = true; break; }
        cal_changed_[u] = root_needs ? 1 : 0;
        continue;
      }
      // Barren node pruning: skip cliques not on path to any query variable.
      if (has_query_scope_ && !query_relevant_[u]) continue;

      std::size_t p = parent_of_[u];
      bool needs = ev_clique_[p] || down_changed_[p];
      for (std::size_t s : children_of_[p]) if (s != u && up_changed_[s]) { needs = true; break; }
      if (!needs && !is_first_calibration_) {
        std::copy(base_down_buf_[u].begin(), base_down_buf_[u].end(), down_msg_buf_[u].begin());
      } else {
        const DistributeOp &op = distribute_ops_[u];
        const double *ppot = clique_potentials_[p].tensor().data();
        const std::size_t PS = op.product_states;
        for (std::size_t pi = 0; pi < PS; ++pi) scratch[pi] = ppot[op.pot_map[pi]];
        if (op.has_parent_down) {
          const double *pm = down_msg_buf_[p].data();
          const uint32_t *dm = op.down_map.data();
          for (std::size_t pi = 0; pi < PS; ++pi) scratch[pi] *= pm[dm[pi]];
        }
        for (std::size_t si = 0; si < op.sibling_cliques.size(); ++si) {
          const double *sm = up_msg_buf_[op.sibling_cliques[si]].data();
          const uint32_t *smap = op.sibling_maps[si].data();
          for (std::size_t pi = 0; pi < PS; ++pi) scratch[pi] *= sm[smap[pi]];
        }
        double *dmsg = down_msg_buf_[u].data();
        const std::size_t DS = down_msg_size_[u];
        std::fill(dmsg, dmsg + DS, 0.0);
        const uint32_t *mm = op.marg_map.data();
        for (std::size_t pi = 0; pi < PS; ++pi) dmsg[mm[pi]] += scratch[pi];
        down_changed_[u] = 1;
      }
      cal_changed_[u] = (ev_clique_[u] || down_changed_[u]) ? 1 : 0;
      for (std::size_t c : children_of_[u]) if (up_changed_[c]) { cal_changed_[u] = 1; break; }
    }

    // ── Assemble calibrated potentials ─────────────────────────────────────
    for (std::size_t i = 0; i < n; ++i) {
      // Barren node pruning: only assemble cliques containing query variables.
      if (has_query_scope_ && !is_query_clique_[i]) continue;

      if (!cal_changed_[i] && !is_first_calibration_) {
        std::copy(base_cal_buf_[i].begin(), base_cal_buf_[i].end(), cal_pot_buf_[i].begin());
        continue;
      }
      const AssembleOp &op = assemble_ops_[i];
      const double *pot = clique_potentials_[i].tensor().data();
      double *out = cal_pot_buf_[i].data();
      const std::size_t NS = op.product_states;
      std::copy(pot, pot + NS, out);
      if (op.has_down) {
        const double *dm = down_msg_buf_[i].data();
        const uint32_t *dmap = op.down_map.data();
        for (std::size_t pi = 0; pi < NS; ++pi) out[pi] *= dm[dmap[pi]];
      }
      for (std::size_t ci = 0; ci < op.child_cliques.size(); ++ci) {
        const double *cm = up_msg_buf_[op.child_cliques[ci]].data();
        const uint32_t *cmap = op.child_maps[ci].data();
        for (std::size_t pi = 0; pi < NS; ++pi) out[pi] *= cm[cmap[pi]];
      }
    }
  } else {
    // ── Collect: leaves → root (batched) ───────────────────────────────────
    for (std::size_t u : collect_order_) {
      if (parent_of_[u] == ~std::size_t(0)) continue;
      bool needs = ev_clique_[u];
      for (std::size_t c : children_of_[u]) if (up_changed_[c]) { needs = true; break; }
      if (!needs && !is_first_calibration_) {
        std::copy(base_up_buf_[u].begin(), base_up_buf_[u].end(), up_msg_buf_[u].begin());
        continue;
      }

      const CollectOp &op = collect_ops_[u];
      const double *pot = clique_potentials_[u].tensor().data();
      const std::size_t PS = op.product_states;

      for (std::size_t pi = 0; pi < PS; ++pi) {
        double *dst = scratch + pi * B;
        const double *src = pot + static_cast<std::size_t>(op.pot_map[pi]) * B;
#pragma omp simd
        for (std::size_t b = 0; b < B; ++b) {
          dst[b] = src[b];
        }
      }

      for (std::size_t ci = 0; ci < op.child_cliques.size(); ++ci) {
        const double *cm = up_msg_buf_[op.child_cliques[ci]].data();
        const uint32_t *cmap = op.child_maps[ci].data();
        for (std::size_t pi = 0; pi < PS; ++pi) {
          double *dst = scratch + pi * B;
          const double *src = cm + static_cast<std::size_t>(cmap[pi]) * B;
#pragma omp simd
          for (std::size_t b = 0; b < B; ++b) {
            dst[b] *= src[b];
          }
        }
      }

      double *umsg = up_msg_buf_[u].data();
      std::fill(umsg, umsg + up_msg_size_[u], 0.0);
      const uint32_t *mm = op.marg_map.data();
      for (std::size_t pi = 0; pi < PS; ++pi) {
        double *dst = umsg + static_cast<std::size_t>(mm[pi]) * B;
        const double *src = scratch + pi * B;
#pragma omp simd
        for (std::size_t b = 0; b < B; ++b) {
          dst[b] += src[b];
        }
      }
      up_changed_[u] = 1;
    }

    // ── Distribute: root → leaves (batched) ────────────────────────────────
    for (std::size_t u : distribute_order_) {
      if (parent_of_[u] == ~std::size_t(0)) {
        bool root_needs = ev_clique_[u];
        for (std::size_t c : children_of_[u]) if (up_changed_[c]) { root_needs = true; break; }
        cal_changed_[u] = root_needs ? 1 : 0;
        continue;
      }
      // Barren node pruning: skip cliques not on path to any query variable.
      if (has_query_scope_ && !query_relevant_[u]) continue;

      std::size_t p = parent_of_[u];
      bool needs = ev_clique_[p] || down_changed_[p];
      for (std::size_t s : children_of_[p]) if (s != u && up_changed_[s]) { needs = true; break; }
      if (!needs && !is_first_calibration_) {
        std::copy(base_down_buf_[u].begin(), base_down_buf_[u].end(), down_msg_buf_[u].begin());
      } else {
        const DistributeOp &op = distribute_ops_[u];
        const double *ppot = clique_potentials_[p].tensor().data();
        const std::size_t PS = op.product_states;

        for (std::size_t pi = 0; pi < PS; ++pi) {
          double *dst = scratch + pi * B;
          const double *src =
              ppot + static_cast<std::size_t>(op.pot_map[pi]) * B;
#pragma omp simd
          for (std::size_t b = 0; b < B; ++b) {
            dst[b] = src[b];
          }
        }

        if (op.has_parent_down) {
          const double *pm = down_msg_buf_[p].data();
          const uint32_t *dm = op.down_map.data();
          for (std::size_t pi = 0; pi < PS; ++pi) {
            double *dst = scratch + pi * B;
            const double *src = pm + static_cast<std::size_t>(dm[pi]) * B;
#pragma omp simd
            for (std::size_t b = 0; b < B; ++b) {
              dst[b] *= src[b];
            }
          }
        }

        for (std::size_t si = 0; si < op.sibling_cliques.size(); ++si) {
          const double *sm = up_msg_buf_[op.sibling_cliques[si]].data();
          const uint32_t *smap = op.sibling_maps[si].data();
          for (std::size_t pi = 0; pi < PS; ++pi) {
            double *dst = scratch + pi * B;
            const double *src = sm + static_cast<std::size_t>(smap[pi]) * B;
#pragma omp simd
            for (std::size_t b = 0; b < B; ++b) {
              dst[b] *= src[b];
            }
          }
        }

        double *dmsg = down_msg_buf_[u].data();
        std::fill(dmsg, dmsg + down_msg_size_[u], 0.0);
        const uint32_t *mm = op.marg_map.data();
        for (std::size_t pi = 0; pi < PS; ++pi) {
          double *dst = dmsg + static_cast<std::size_t>(mm[pi]) * B;
          const double *src = scratch + pi * B;
#pragma omp simd
          for (std::size_t b = 0; b < B; ++b) {
            dst[b] += src[b];
          }
        }
        down_changed_[u] = 1;
      }

      cal_changed_[u] = (ev_clique_[u] || down_changed_[u]) ? 1 : 0;
      for (std::size_t c : children_of_[u]) if (up_changed_[c]) { cal_changed_[u] = 1; break; }
    }

    // ── Assemble calibrated potentials (batched) ───────────────────────────
    for (std::size_t i = 0; i < n; ++i) {
      // Barren node pruning: only assemble cliques containing query variables.
      if (has_query_scope_ && !is_query_clique_[i]) continue;

      if (!cal_changed_[i] && !is_first_calibration_) {
        std::copy(base_cal_buf_[i].begin(), base_cal_buf_[i].end(), cal_pot_buf_[i].begin());
        continue;
      }

      const AssembleOp &op = assemble_ops_[i];
      const std::size_t NS = op.product_states;
      const double *pot = clique_potentials_[i].tensor().data();
      double *out = cal_pot_buf_[i].data();
      std::copy(pot, pot + (NS * B), out);

      if (op.has_down) {
        const double *dm = down_msg_buf_[i].data();
        const uint32_t *dmap = op.down_map.data();
        for (std::size_t pi = 0; pi < NS; ++pi) {
          double *dst = out + pi * B;
          const double *src = dm + static_cast<std::size_t>(dmap[pi]) * B;
#pragma omp simd
          for (std::size_t b = 0; b < B; ++b) {
            dst[b] *= src[b];
          }
        }
      }

      for (std::size_t ci = 0; ci < op.child_cliques.size(); ++ci) {
        const double *cm = up_msg_buf_[op.child_cliques[ci]].data();
        const uint32_t *cmap = op.child_maps[ci].data();
        for (std::size_t pi = 0; pi < NS; ++pi) {
          double *dst = out + pi * B;
          const double *src = cm + static_cast<std::size_t>(cmap[pi]) * B;
#pragma omp simd
          for (std::size_t b = 0; b < B; ++b) {
            dst[b] *= src[b];
          }
        }
      }
    }
  }

  // ── Global inconsistency check ────────────────────────────────────────────
  if (root_clique_ != ~std::size_t(0)) {
    double total = 0.0;
    bool has_nan = false;
    for (double v : cal_pot_buf_[root_clique_]) {
      if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
      total += v;
    }
    if (has_nan) {
      throw std::invalid_argument(
          "Numerical instability detected: NaN/Inf in calibrated potential. "
          "This may indicate underflow in a deep network. Consider using "
          "log-space calibration or checking CPT values.");
    }
    if (evidence_matrix_ && B == 1 && total <= 0.0) {
      throw std::invalid_argument("Inconsistent evidence");
    }
  }

  // ── Base snapshot on first calibration ───────────────────────────────────
  if (is_first_calibration_) {
    for (std::size_t u = 0; u < n; ++u) {
      if (parent_of_[u] == ~std::size_t(0)) continue;
      base_up_buf_[u]   = up_msg_buf_[u];
      base_down_buf_[u] = down_msg_buf_[u];
    }
    for (std::size_t i = 0; i < n; ++i) base_cal_buf_[i] = cal_pot_buf_[i];
    is_first_calibration_ = false;
  }
}


// ---------------------------------------------------------------------------
// query_marginals_multi — per-query extraction using pre-baked state_of_dim.
//
// batch_size == 1 (HCL point mode): queries are grouped by clique so each
//   unique clique is scanned exactly once for all of its query variables.
//   Zero heap allocation; uses uint8 state_of_dim and workspace-owned flags.
// batch_size > 1: direct extraction from calibrated batched beliefs.
// ---------------------------------------------------------------------------
void BatchWorkspace::query_marginals_multi(const NodeId *query_vars,
                                           std::size_t   num_queries,
                                           const std::size_t *offsets,
                                           double *out) const {
  if (num_queries == 0) {
    return;
  }

  struct QEntry {
    std::size_t qi;
    std::size_t node_dim;
    std::size_t n_states;
  };

  constexpr std::size_t kStackQueryLimit = 64;
  constexpr std::size_t kStackCliqueLimit = 200;

  std::size_t clique_of_q_stack[kStackQueryLimit];
  QEntry entries_stack[kStackQueryLimit];
  uint8_t visited_stack[kStackCliqueLimit];

  std::vector<std::size_t> clique_of_q_heap;
  std::vector<QEntry> entries_heap;

  std::size_t *clique_of_q = nullptr;
  QEntry *entries = nullptr;
  uint8_t *visited = nullptr;

  if (num_queries <= kStackQueryLimit) {
    clique_of_q = clique_of_q_stack;
    entries = entries_stack;
  } else {
    clique_of_q_heap.resize(num_queries);
    entries_heap.resize(num_queries);
    clique_of_q = clique_of_q_heap.data();
    entries = entries_heap.data();
  }

  const std::size_t n_cliques = jt_.cliques().size();
  if (n_cliques <= kStackCliqueLimit) {
    std::memset(visited_stack, 0, n_cliques);
    visited = visited_stack;
  } else {
    if (query_visited_scratch_.size() < n_cliques) {
      query_visited_scratch_.resize(n_cliques);
    }
    std::memset(query_visited_scratch_.data(), 0, n_cliques);
    visited = query_visited_scratch_.data();
  }

  const std::size_t B = batch_size_;
  const std::size_t total_states = offsets[num_queries];

  // First pass: resolve cliques and node dimensions + zero-fill output slices.
  for (std::size_t qi = 0; qi < num_queries; ++qi) {
    NodeId nid = query_vars[qi];
    auto it = node_to_clique_.find(nid);
    if (it == node_to_clique_.end())
      throw std::invalid_argument("query node not in JT");
    std::size_t ci = it->second;
    clique_of_q[qi] = ci;
    const auto &scope = jt_.cliques()[ci].scope;
    std::size_t node_dim = 0;
    for (std::size_t d = 0; d < scope.size(); ++d) {
      if (scope[d] == nid) {
        node_dim = d;
        break;
      }
    }

    entries[qi] = {qi, node_dim, jt_.graph()->get_variable(nid).states.size()};

    if (B == 1) {
      double *qout = out + offsets[qi];
      std::fill(qout, qout + entries[qi].n_states, 0.0);
    } else {
      for (std::size_t b = 0; b < B; ++b) {
        double *qout = out + b * total_states + offsets[qi];
        std::fill(qout, qout + entries[qi].n_states, 0.0);
      }
    }
  }

  // Second pass: process each clique exactly once.
  for (std::size_t qi = 0; qi < num_queries; ++qi) {
    std::size_t ci = clique_of_q[qi];
    if (visited[ci]) continue;
    visited[ci] = 1;

    const CliqueMargInfo &mi = clique_marg_info_[ci];
    const double *cal = cal_pot_buf_[ci].data();
    const std::size_t total = mi.n_states_total;

    if (B == 1) {
      for (std::size_t flat = 0; flat < total; ++flat) {
        const double v = cal[flat];
        for (std::size_t qj = qi; qj < num_queries; ++qj) {
          if (clique_of_q[qj] != ci) continue;
          const QEntry &e = entries[qj];
          out[offsets[e.qi] + mi.state_of_dim[e.node_dim][flat]] += v;
        }
      }

      for (std::size_t qj = qi; qj < num_queries; ++qj) {
        if (clique_of_q[qj] != ci) continue;
        const QEntry &e = entries[qj];
        double *qout = out + offsets[e.qi];
        double sum = 0.0;
        for (std::size_t s = 0; s < e.n_states; ++s) sum += qout[s];
        if (sum > 0.0) {
          for (std::size_t s = 0; s < e.n_states; ++s) qout[s] /= sum;
        }
      }
    } else {
      for (std::size_t flat = 0; flat < total; ++flat) {
        const double *cal_row = cal + flat * B;
        for (std::size_t qj = qi; qj < num_queries; ++qj) {
          if (clique_of_q[qj] != ci) continue;
          const QEntry &e = entries[qj];
          const std::size_t state = mi.state_of_dim[e.node_dim][flat];
          for (std::size_t b = 0; b < B; ++b) {
            out[b * total_states + offsets[e.qi] + state] += cal_row[b];
          }
        }
      }

      for (std::size_t qj = qi; qj < num_queries; ++qj) {
        if (clique_of_q[qj] != ci) continue;
        const QEntry &e = entries[qj];
        for (std::size_t b = 0; b < B; ++b) {
          double *qout = out + b * total_states + offsets[e.qi];
          double sum = 0.0;
          for (std::size_t s = 0; s < e.n_states; ++s) sum += qout[s];
          if (sum > 0.0) {
            for (std::size_t s = 0; s < e.n_states; ++s) qout[s] /= sum;
          } else {
            std::fill(qout, qout + e.n_states, 0.0);
          }
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// query_marginal — backward-compat wrapper (calls query_marginals_multi)
// ---------------------------------------------------------------------------
DenseTensor BatchWorkspace::query_marginal(NodeId node) const {
  std::size_t n_states = jt_.graph()->get_variable(node).states.size();
  std::vector<std::size_t> out_shape = {batch_size_, n_states};
  DenseTensor result(out_shape);
  result.fill(0.0);

  std::size_t offsets[2] = {0, n_states};
  query_marginals_multi(&node, 1, offsets, result.data());
  return result;
}


// ---------------------------------------------------------------------------
// max_calibrate — max-product belief propagation for MAP / MPE inference
//
// Semantics: replaces the sum-marginalization in the collect pass with max.
// After max_calibrate(), call query_map() to decode the joint MAP assignment.
//
// Evidence is applied exactly as in calibrate() — the same
// apply_soft_evidence_to_clique() helper is used, followed by a max-product
// collect + assemble pass.
//
// Result is stored in map_cal_pot_buf_, which is separate from cal_pot_buf_
// so that sum-product and max-product results can coexist.
// ---------------------------------------------------------------------------
void BatchWorkspace::max_calibrate() {
  const std::size_t n = jt_.cliques().size();
  if (batch_size_ != 1) {
    throw std::invalid_argument(
        "BatchWorkspace::max_calibrate currently supports batch_size == 1. "
        "Use BatchExecutionEngine::evaluate_map for batched MAP.");
  }

  // Allocate max-product buffers on first use (same layout as sum-product).
  if (map_up_msg_buf_.size() != n) {
    map_up_msg_buf_.resize(n);
    map_cal_pot_buf_.resize(n);
    map_traceback_.resize(n);
    for (std::size_t u = 0; u < n; ++u) {
      if (parent_of_[u] == ~std::size_t(0)) continue;
      map_up_msg_buf_[u].assign(sepset_size_[u], 0.0);
      map_traceback_[u].assign(collect_ops_[u].product_states, 0);
    }
    for (std::size_t i = 0; i < n; ++i)
      map_cal_pot_buf_[i].assign(clique_states_[i], 0.0);
  }

  // Restore base clique potentials (start from prior).
  if (!base_clique_potentials_.empty()) {
    clique_potentials_.clear();
    for (const Factor &bp : base_clique_potentials_)
      clique_potentials_.push_back(
          Factor(bp.scope(), bp.tensor().shape(),
                 const_cast<double *>(bp.tensor().data())));
  }
  std::fill(ev_scratch_active_.begin(), ev_scratch_active_.end(), false);
  std::memset(ev_clique_.data(), 0, n);

  // Apply evidence.
  if (evidence_matrix_ || !soft_evidence_scalar_.empty() ||
      !soft_evidence_matrix_.empty()) {
    for (std::size_t ci = 0; ci < n; ++ci)
      apply_soft_evidence_to_clique(ci);
  }

  double *scratch = scratch_buf_.data();

  // ── Max-product collect pass (B=1 only — MAP is always per-row) ─────────
  for (std::size_t u : collect_order_) {
    if (parent_of_[u] == ~std::size_t(0)) continue;
    const CollectOp &op  = collect_ops_[u];
    const double    *pot = clique_potentials_[u].tensor().data();
    const std::size_t PS = op.product_states;

    // Gather clique potential into scratch.
    for (std::size_t pi = 0; pi < PS; ++pi)
      scratch[pi] = pot[op.pot_map[pi]];

    // Multiply incoming child up-messages.
    for (std::size_t ci = 0; ci < op.child_cliques.size(); ++ci) {
      const double    *cm   = map_up_msg_buf_[op.child_cliques[ci]].data();
      const uint32_t  *cmap = op.child_maps[ci].data();
      for (std::size_t pi = 0; pi < PS; ++pi) scratch[pi] *= cm[cmap[pi]];
    }

    // Max-marginalize into up-message (replace scatter-add with scatter-max).
    double *umsg = map_up_msg_buf_[u].data();
    const std::size_t US = sepset_size_[u];
    std::fill(umsg, umsg + US, 0.0);
    const uint32_t *mm = op.marg_map.data();
    for (std::size_t pi = 0; pi < PS; ++pi) {
      if (scratch[pi] > umsg[mm[pi]]) umsg[mm[pi]] = scratch[pi];
    }
  }

  // ── Assemble max-product calibrated beliefs at root ──────────────────────
  // For MAP we only need the root clique to extract the joint argmax.
  // For MPE over all variables, we assemble every clique.
  for (std::size_t i = 0; i < n; ++i) {
    const AssembleOp &op  = assemble_ops_[i];
    const double     *pot = clique_potentials_[i].tensor().data();
    double           *out = map_cal_pot_buf_[i].data();
    const std::size_t NS  = op.product_states;
    std::copy(pot, pot + NS, out);
    // Multiply incoming child up-messages from the max-product pass.
    for (std::size_t ci = 0; ci < op.child_cliques.size(); ++ci) {
      const double   *cm   = map_up_msg_buf_[op.child_cliques[ci]].data();
      const uint32_t *cmap = op.child_maps[ci].data();
      for (std::size_t pi = 0; pi < NS; ++pi) out[pi] *= cm[cmap[pi]];
    }
    // Note: in max-product, the root assembler ignores down-messages (no parent).
    // Non-root cliques do not receive down-messages in this simplified implementation;
    // the root clique's calibrated beliefs are used for argmax.
  }
}

// ---------------------------------------------------------------------------
// query_map — decode MAP assignment from max-product calibrated beliefs
//
// out_states[v] = argmax state for variable v, or -1 if v is not in the JT.
// Caller must allocate out_states[n_vars].
// ---------------------------------------------------------------------------
void BatchWorkspace::query_map(int *out_states, std::size_t n_vars) const {
  // Initialise all states to -1 (not in JT / not processed).
  std::fill(out_states, out_states + n_vars, -1);

  const std::size_t n = jt_.cliques().size();

  // For each variable, find the clique that contains it and extract its
  // marginal argmax from the max-product calibrated beliefs.
  for (const auto &[nid, ci] : node_to_clique_) {
    if (nid >= static_cast<NodeId>(n_vars)) continue;

    const CliqueMargInfo &mi  = clique_marg_info_[ci];
    const double         *cal = map_cal_pot_buf_[ci].data();
    const std::size_t     tot = mi.n_states_total;

    // Find the dimension index of this variable in the clique scope.
    const auto &scope = jt_.cliques()[ci].scope;
    std::size_t node_dim = 0;
    for (std::size_t d = 0; d < scope.size(); ++d)
      if (scope[d] == nid) { node_dim = d; break; }

    const std::size_t n_states = jt_.graph()->get_variable(nid).states.size();

    // Accumulate the max-product beliefs for each state (marginal max).
    double best_val = -1.0;
    int    best_s   = 0;
    for (std::size_t s = 0; s < n_states; ++s) {
      double v = 0.0;
      for (std::size_t flat = 0; flat < tot; ++flat)
        if (mi.state_of_dim[node_dim][flat] == static_cast<uint8_t>(s))
          v = std::max(v, cal[flat]);
      if (v > best_val) { best_val = v; best_s = static_cast<int>(s); }
    }
    out_states[nid] = best_s;
  }
}

} // namespace bncore
