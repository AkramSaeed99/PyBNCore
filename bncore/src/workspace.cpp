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
  for (std::size_t u = 0; u < n; ++u) {
    if (parent_of_[u] == ~std::size_t(0)) continue;
    up_msg_buf_[u].assign(sepset_size_[u], 0.0);
    down_msg_buf_[u].assign(sepset_size_[u], 0.0);
    up_msg_size_[u] = down_msg_size_[u] = sepset_size_[u];
    base_up_buf_[u].assign(sepset_size_[u], 0.0);
    base_down_buf_[u].assign(sepset_size_[u], 0.0);
  }

  // Calibrated potential buffers
  cal_pot_buf_.resize(n);  cal_pot_size_.resize(n);  base_cal_buf_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    cal_pot_buf_[i].assign(clique_states_[i], 0.0);
    cal_pot_size_[i] = clique_states_[i];
    base_cal_buf_[i].assign(clique_states_[i], 0.0);
  }

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
  // This is construction-time only — ok to use large temporary allocator.
  // We use a local temporary std::vector-backed buffer so this is safe for
  // any target_batch_size.
  clique_potentials_.clear();
  clique_potentials_.reserve(jt_.cliques().size());

  for (const auto &clique : jt_.cliques()) {
    const Factor &unbatched = clique.base_potential;
    std::vector<std::size_t> new_sizes;
    for (std::size_t i = 0; i < unbatched.scope().size(); ++i)
      new_sizes.push_back(unbatched.tensor().shape()[i]);
    new_sizes.push_back(target_batch_size);
    std::size_t num_states = unbatched.tensor().size();

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
      double *d = cpt.tensor().data();
      if (probs.size() == fam_states * target_batch_size) {
        std::copy(probs.begin(), probs.end(), d);
      } else {
        for (std::size_t s = 0; s < fam_states; ++s)
          for (std::size_t b = 0; b < target_batch_size; ++b)
            d[s * target_batch_size + b] = probs[s];
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
  is_first_calibration_ = true;
  clear_evidence();
  calibrate();
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------
void BatchWorkspace::reset(std::size_t new_batch_size) {
  batch_size_ = new_batch_size;
  clear_evidence();

  // Restore evidence scratch tracking
  std::fill(ev_scratch_active_.begin(), ev_scratch_active_.end(), false);

  if (batch_size_ == 1 && !base_clique_potentials_.empty()) {
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
// Evidence
// ---------------------------------------------------------------------------
void BatchWorkspace::set_evidence_matrix(const int *ev, std::size_t num_vars) {
  evidence_matrix_ = ev; evidence_num_vars_ = num_vars;
}
void BatchWorkspace::clear_evidence() {
  evidence_matrix_ = nullptr; evidence_num_vars_ = 0;
}

// ---------------------------------------------------------------------------
// calibrate — ZERO heap allocation in the hot path
// ---------------------------------------------------------------------------
void BatchWorkspace::calibrate() {
  std::size_t n = jt_.cliques().size();
  std::vector<bool> ev_clique(n, false);

  // ── P3: Evidence application — no fixed-size allocator cap ──────────────
  if (evidence_matrix_ && batch_size_ == 1) {
    const int *ev = evidence_matrix_;

    // Walk only the evidence nodes, not all cliques
    for (std::size_t vid = 0; vid < evidence_num_vars_; ++vid) {
      if (ev[vid] < 0) continue;
      auto it = node_in_cliques_.find((NodeId)vid);
      if (it == node_in_cliques_.end()) continue;

      for (std::size_t ci : it->second) {
        if (ev_clique[ci]) continue;

        const Factor &pot = clique_potentials_[ci];
        const auto &scope = pot.scope();
        const auto &shape = pot.tensor().shape();
        std::size_t K = scope.size();

        // Collect all evidence constraints for this clique (stack-local)
        struct EV { std::size_t pos, states; int obs; };
        EV ev_vars[32]; std::size_t nev = 0;
        for (std::size_t d = 0; d < K; ++d) {
          NodeId nid = scope[d];
          if (nid < evidence_num_vars_ && ev[nid] >= 0)
            ev_vars[nev++] = {d, shape[d], ev[nid]};
        }
        if (nev == 0) continue;

        // Strides (stack)
        std::size_t strides[32];
        strides[K - 1] = 1;
        for (int d = (int)K - 2; d >= 0; --d)
          strides[d] = strides[d + 1] * shape[d + 1];

        // P3: lazily size per-clique scratch buffer
        std::size_t total = clique_states_[ci];
        std::vector<double> &scratch = ev_scratch_[ci];
        scratch.resize(total);
        const double *src = pot.tensor().data();
        std::copy(src, src + total, scratch.data());

        double sum = 0.0;
        for (std::size_t idx = 0; idx < total; ++idx) {
          bool zero = false;
          std::size_t rem = idx;
          for (std::size_t ei = 0; ei < nev; ++ei) {
            std::size_t coord = (rem / strides[ev_vars[ei].pos]) % ev_vars[ei].states;
            if ((int)coord != ev_vars[ei].obs) { zero = true; break; }
          }
          if (zero) scratch[idx] = 0.0; else sum += scratch[idx];
        }
        if (sum == 0.0) throw std::invalid_argument("Inconsistent evidence");

        // Rebind the clique potential to the scratch
        clique_potentials_[ci] = Factor(scope, shape, scratch.data());
        ev_scratch_active_[ci] = true;
        ev_clique[ci] = true;
      }
    }
  } else if (evidence_matrix_ && batch_size_ > 1) {
    // Batched path — rare, use generic Factor ops
    for (std::size_t i = 0; i < n; ++i) {
      for (NodeId nid : clique_potentials_[i].scope()) {
        if (nid >= evidence_num_vars_) continue;
        bool has_ev = false;
        for (std::size_t b = 0; b < batch_size_; ++b)
          if (evidence_matrix_[b * evidence_num_vars_ + nid] != -1) { has_ev = true; break; }
        if (!has_ev) continue;
        std::size_t ns = jt_.graph()->get_variable(nid).states.size();
        std::vector<NodeId> ind_scope = {nid};
        std::vector<std::size_t> ind_sh = {ns, batch_size_};
        Factor ind(ind_scope, ind_sh);
        for (std::size_t s = 0; s < ns; ++s)
          for (std::size_t b = 0; b < batch_size_; ++b) {
            int v = evidence_matrix_[b * evidence_num_vars_ + nid];
            ind.tensor().data()[s * batch_size_ + b] = (v < 0 || v == (int)s) ? 1.0 : 0.0;
          }
        clique_potentials_[i] = clique_potentials_[i].multiply(ind);
        ev_clique[i] = true;
      }
    }
  }

  // ── Collect: leaves → root ───────────────────────────────────────────────
  std::vector<bool> up_changed(n, false);
  double *scratch = scratch_buf_.data();  // P1: reuse persistent scratch

  for (std::size_t u : collect_order_) {
    if (parent_of_[u] == ~std::size_t(0)) continue;

    bool needs = ev_clique[u];
    for (std::size_t c : children_of_[u]) if (up_changed[c]) { needs = true; break; }

    if (!needs && !is_first_calibration_) {
      std::copy(base_up_buf_[u].begin(), base_up_buf_[u].end(), up_msg_buf_[u].begin());
      continue;
    }

    const CollectOp &op = collect_ops_[u];
    const double *pot = clique_potentials_[u].tensor().data();
    const std::size_t PS = op.product_states;

    // P1: write into persistent scratch — no heap alloc
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
    up_changed[u] = true;
  }

  // ── Distribute: root → leaves ────────────────────────────────────────────
  std::vector<bool> down_changed(n, false);
  std::vector<bool> cal_changed(n, false);

  for (std::size_t u : distribute_order_) {
    if (parent_of_[u] == ~std::size_t(0)) {
      bool root_needs = ev_clique[u];
      for (std::size_t c : children_of_[u]) if (up_changed[c]) { root_needs = true; break; }
      cal_changed[u] = root_needs;
      continue;
    }

    std::size_t p = parent_of_[u];
    bool needs = ev_clique[p] || down_changed[p];
    for (std::size_t s : children_of_[p]) if (s != u && up_changed[s]) { needs = true; break; }

    if (!needs && !is_first_calibration_) {
      std::copy(base_down_buf_[u].begin(), base_down_buf_[u].end(), down_msg_buf_[u].begin());
    } else {
      const DistributeOp &op = distribute_ops_[u];
      const double *ppot = clique_potentials_[p].tensor().data();
      const std::size_t PS = op.product_states;

      // P1: write into persistent scratch
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
      down_changed[u] = true;
    }

    cal_changed[u] = ev_clique[u] || down_changed[u];
    for (std::size_t c : children_of_[u]) if (up_changed[c]) { cal_changed[u] = true; break; }
  }

  // ── Assemble calibrated potentials ───────────────────────────────────────
  for (std::size_t i = 0; i < n; ++i) {
    if (!cal_changed[i] && !is_first_calibration_) {
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

  // ── Global inconsistency check ────────────────────────────────────────────
  if (evidence_matrix_ && batch_size_ == 1 && root_clique_ != ~std::size_t(0)) {
    double total = 0.0;
    for (double v : cal_pot_buf_[root_clique_]) total += v;
    if (total <= 0.0) throw std::invalid_argument("Inconsistent evidence");
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
// batch_size == 1 (HCL point mode): zero-allocation, uses uint8 state_of_dim.
// batch_size > 1: correct fallback through Factor::marginalize (the proven
//   safe path). cal_pot_buf_ is sized as [clique_states] not
//   [clique_states × batch_size], so the interleaved read that was here
//   before was incorrect and is removed.
// ---------------------------------------------------------------------------
void BatchWorkspace::query_marginals_multi(const NodeId *query_vars,
                                           std::size_t   num_queries,
                                           const std::size_t *offsets,
                                           double *out) const {
  for (std::size_t qi = 0; qi < num_queries; ++qi) {
    NodeId nid = query_vars[qi];
    auto it = node_to_clique_.find(nid);
    if (it == node_to_clique_.end())
      throw std::invalid_argument("query node not in JT");

    std::size_t ci = it->second;
    const auto &scope = jt_.cliques()[ci].scope;
    std::size_t n_states = jt_.graph()->get_variable(nid).states.size();

    if (batch_size_ == 1) {
      // Fast path: pre-baked state_of_dim, zero heap allocation.
      const CliqueMargInfo &mi = clique_marg_info_[ci];
      const double *cal = cal_pot_buf_[ci].data();
      const std::size_t total = mi.n_states_total;

      // Find dimension of nid in scope
      std::size_t node_dim = 0;
      for (std::size_t d = 0; d < scope.size(); ++d)
        if (scope[d] == nid) { node_dim = d; break; }

      double *qout = out + offsets[qi];
      std::fill(qout, qout + n_states, 0.0);
      const uint8_t *sod = mi.state_of_dim[node_dim].data();
      for (std::size_t flat = 0; flat < total; ++flat)
        qout[sod[flat]] += cal[flat];
      double sum = 0.0;
      for (std::size_t s = 0; s < n_states; ++s) sum += qout[s];
      if (sum > 0.0) for (std::size_t s = 0; s < n_states; ++s) qout[s] /= sum;

    } else {
      // Correct batched fallback: build a Factor view over cal_pot_buf_ and
      // use Factor::marginalize. cal_pot_buf_ is laid out as un-batched
      // [clique_states] scalars (assemble writes NS = clique_states_ values,
      // not NS * batch_size). For batch_size > 1 the calibrated potentials
      // are stored in the clique_potentials_[ci] tensor which IS batched.
      // Read from there directly.
      const Factor &cpot = clique_potentials_[ci];

      // Build the marginalisation variable list (everything except nid)
      std::vector<NodeId> marg_vars;
      for (NodeId nd : scope) if (nd != nid) marg_vars.push_back(nd);

      Factor marg = cpot.marginalize(marg_vars);
      // marg.tensor() shape: {n_states, batch_size_}  (or {n_states} if bs=1)
      const double *mdata = marg.tensor().data();
      std::size_t bs = batch_size_;

      for (std::size_t b = 0; b < bs; ++b) {
        double sum = 0.0;
        for (std::size_t s = 0; s < n_states; ++s)
          sum += mdata[s * bs + b];
        double *qout = out + b * offsets[num_queries] + offsets[qi];
        for (std::size_t s = 0; s < n_states; ++s)
          qout[s] = (sum > 0.0) ? mdata[s * bs + b] / sum : 0.0;
      }
    }
  }
}


// ---------------------------------------------------------------------------
// query_marginal — backward-compat wrapper (calls query_marginals_multi)
// ---------------------------------------------------------------------------
DenseTensor BatchWorkspace::query_marginal(NodeId node) const {
  std::size_t n_states = jt_.graph()->get_variable(node).states.size();
  std::vector<std::size_t> out_shape = {n_states, batch_size_};
  DenseTensor result(out_shape);
  result.fill(0.0);

  std::size_t offsets[2] = {0, n_states};
  query_marginals_multi(&node, 1, offsets, result.data());
  return result;
}

} // namespace bncore
