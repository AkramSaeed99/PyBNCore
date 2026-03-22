#include "bncore/inference/workspace.hpp"

namespace bncore {

BatchWorkspace::BatchWorkspace(const JunctionTree &jt, std::size_t batch_size)
    : jt_(jt), batch_size_(batch_size), allocator_(1024 * 1024 * 500) {
  rebuild_clique_potentials(batch_size_);

  std::size_t n = jt_.cliques().size();
  root_clique_ = std::numeric_limits<std::size_t>::max();
  parent_of_.assign(n, std::numeric_limits<std::size_t>::max());
  children_of_.assign(n, {});
  distribute_order_.clear();
  collect_order_.clear();

  std::vector<bool> visited(n, false);
  std::vector<std::size_t> stack;
  for (std::size_t start = 0; start < n; ++start) {
    if (visited[start]) {
      continue;
    }
    if (root_clique_ == std::numeric_limits<std::size_t>::max()) {
      root_clique_ = start;
    }
    stack.push_back(start);
    visited[start] = true;

    while (!stack.empty()) {
      std::size_t u = stack.back();
      stack.pop_back();
      distribute_order_.push_back(u);

      for (std::size_t v : jt_.cliques()[u].neighbor_cliques) {
        if (!visited[v]) {
          visited[v] = true;
          parent_of_[v] = u;
          children_of_[u].push_back(v);
          stack.push_back(v);
        }
      }
    }
  }

  collect_order_ = distribute_order_;
  std::reverse(collect_order_.begin(), collect_order_.end());
}

void BatchWorkspace::snapshot_base_potentials() {
  base_clique_potentials_.clear();
  base_clique_potentials_.reserve(clique_potentials_.size());
  for (const Factor &pot : clique_potentials_) {
    const auto &shape = pot.tensor().shape();
    Factor owned_copy(pot.scope(), shape);
    std::copy(pot.tensor().data(), pot.tensor().data() + pot.tensor().size(),
              owned_copy.tensor().data());
    base_clique_potentials_.push_back(std::move(owned_copy));
  }
}

void BatchWorkspace::rebuild_clique_potentials(std::size_t target_batch_size) {
  clique_potentials_.clear();
  clique_potentials_.reserve(jt_.cliques().size());

  for (const auto &clique : jt_.cliques()) {
    const Factor &unbatched = clique.base_potential;
    std::vector<std::size_t> new_sizes;
    for (std::size_t i = 0; i < unbatched.scope().size(); ++i) {
      new_sizes.push_back(unbatched.tensor().shape()[i]);
    }
    new_sizes.push_back(target_batch_size);

    std::size_t num_states = unbatched.tensor().size();
    Factor batched(unbatched.scope(), new_sizes,
                   allocator_.allocate(num_states * target_batch_size));
    batched.tensor().fill(1.0);

    for (NodeId n_id : clique.assigned_cpts) {
      std::vector<NodeId> family;
      for (NodeId p : jt_.graph()->get_parents(n_id))
        family.push_back(p);
      family.push_back(n_id);

      std::vector<std::size_t> fam_sizes;
      for (NodeId p : family)
        fam_sizes.push_back(jt_.graph()->get_variable(p).states.size());

      std::vector<std::size_t> new_fam_sizes = fam_sizes;
      new_fam_sizes.push_back(target_batch_size);

      std::size_t fam_states = 1;
      for (auto s : fam_sizes)
        fam_states *= s;

      Factor cpt_factor(family, new_fam_sizes,
                        allocator_.allocate(fam_states * target_batch_size));

      const auto &probs = jt_.graph()->get_variable(n_id).cpt;
      if (probs.size() == fam_states * target_batch_size) {
        std::copy(probs.begin(), probs.end(), cpt_factor.tensor().data());
      } else if (probs.size() == fam_states) {
        for (std::size_t s = 0; s < fam_states; ++s) {
          double val = probs[s];
          double *b_ptr = cpt_factor.tensor().data() + s * target_batch_size;
          for (std::size_t b = 0; b < target_batch_size; ++b)
            b_ptr[b] = val;
        }
      } else {
        throw std::invalid_argument("BatchWorkspace: CPT geometry bound "
                                    "strictly mismatched for Node ID " +
                                    std::to_string(n_id) + ".");
      }
      batched = batched.multiply(cpt_factor, &allocator_);
    }
    clique_potentials_.push_back(std::move(batched));
  }

  snapshot_base_potentials();
}

void BatchWorkspace::reset(std::size_t new_batch_size) {
  batch_size_ = new_batch_size;
  allocator_.reset();
  clear_evidence();
  if (!base_clique_potentials_.empty()) {
    const auto &shape0 = base_clique_potentials_.front().tensor().shape();
    const std::size_t base_batch_size = shape0.back();
    if (base_batch_size == batch_size_) {
      clique_potentials_.clear();
      clique_potentials_.reserve(base_clique_potentials_.size());
      for (const Factor &base_pot : base_clique_potentials_) {
        const auto &shape = base_pot.tensor().shape();
        Factor copied(base_pot.scope(), shape,
                      allocator_.allocate(base_pot.tensor().size()));
        std::copy(base_pot.tensor().data(),
                  base_pot.tensor().data() + base_pot.tensor().size(),
                  copied.tensor().data());
        clique_potentials_.push_back(std::move(copied));
      }
      return;
    }
  }

  rebuild_clique_potentials(batch_size_);
}

void BatchWorkspace::set_evidence_matrix(const int *evidence_matrix,
                                         std::size_t num_vars) {
  evidence_matrix_ = evidence_matrix;
  evidence_num_vars_ = num_vars;
}

void BatchWorkspace::clear_evidence() {
  evidence_matrix_ = nullptr;
  evidence_num_vars_ = 0;
}

void BatchWorkspace::calibrate() {
  std::size_t n = jt_.cliques().size();

  for (std::size_t i = 0; i < n; ++i) {
    if (!evidence_matrix_)
      break;
    const std::vector<NodeId> scope_nodes = clique_potentials_[i].scope();
    // Apply evidence indicators only for scoped variables that are actually
    // constrained for at least one batch row.
    for (NodeId node_id : scope_nodes) {
      if (node_id >= evidence_num_vars_)
        continue;

      bool has_constrained_evidence = false;
      for (std::size_t b = 0; b < batch_size_; ++b) {
        const int val = evidence_matrix_[b * evidence_num_vars_ + node_id];
        if (val != -1) {
          has_constrained_evidence = true;
          break;
        }
      }
      if (!has_constrained_evidence)
        continue;

      std::size_t num_states = jt_.graph()->get_variable(node_id).states.size();
      Factor indicator({node_id}, {num_states, batch_size_},
                       allocator_.allocate(num_states * batch_size_));
      for (std::size_t s = 0; s < num_states; ++s) {
        double *ptr = indicator.tensor().data() + s * batch_size_;
        for (std::size_t b = 0; b < batch_size_; ++b) {
          int val = evidence_matrix_[b * evidence_num_vars_ + node_id];
          ptr[b] = (val == -1 || val == static_cast<int>(s)) ? 1.0 : 0.0;
        }
      }
      clique_potentials_[i] = clique_potentials_[i].multiply(indicator, &allocator_);
    }
  }

  auto get_marg_vars = [&](std::size_t u, std::size_t neighbor) {
    std::vector<NodeId> sepset;
    std::set_intersection(
        jt_.cliques()[u].scope.begin(), jt_.cliques()[u].scope.end(),
        jt_.cliques()[neighbor].scope.begin(),
        jt_.cliques()[neighbor].scope.end(), std::back_inserter(sepset));
    std::vector<NodeId> marg_vars;
    std::set_difference(jt_.cliques()[u].scope.begin(),
                        jt_.cliques()[u].scope.end(), sepset.begin(),
                        sepset.end(), std::back_inserter(marg_vars));
    return marg_vars;
  };

  // Phase 1: Collect (Leaves to Root)
  up_messages_.clear();
  up_messages_.resize(n, Factor({}, {1}));
  for (std::size_t u : collect_order_) {
    if (parent_of_[u] == std::numeric_limits<std::size_t>::max())
      continue;
    Factor product = clique_potentials_[u];
    for (std::size_t v : children_of_[u])
      product = product.multiply(up_messages_[v], &allocator_);
    up_messages_[u] =
        product.marginalize(get_marg_vars(u, parent_of_[u]), &allocator_);
  }

  // Phase 2: Distribute (Root to Leaves)
  down_messages_.clear();
  down_messages_.resize(n, Factor({}, {1}));
  for (std::size_t u : distribute_order_) {
    if (parent_of_[u] == std::numeric_limits<std::size_t>::max())
      continue;
    std::size_t p = parent_of_[u];
    Factor product = clique_potentials_[p];
    if (parent_of_[p] != std::numeric_limits<std::size_t>::max())
      product = product.multiply(down_messages_[p], &allocator_);
    for (std::size_t sibling : children_of_[p]) {
      if (sibling != u)
        product = product.multiply(up_messages_[sibling], &allocator_);
    }
    down_messages_[u] = product.marginalize(get_marg_vars(p, u), &allocator_);
  }

  calibrated_potentials_.clear();
  calibrated_potentials_.resize(n, Factor({}, {1}));
  for (std::size_t i = 0; i < n; ++i) {
    Factor final_pot = clique_potentials_[i];
    if (parent_of_[i] != std::numeric_limits<std::size_t>::max())
      final_pot = final_pot.multiply(down_messages_[i], &allocator_);
    for (std::size_t child : children_of_[i])
      final_pot = final_pot.multiply(up_messages_[child], &allocator_);
    calibrated_potentials_[i] = final_pot;
  }
}

DenseTensor BatchWorkspace::query_marginal(NodeId node) const {
  for (std::size_t i = 0; i < calibrated_potentials_.size(); ++i) {
    const Factor &pot = calibrated_potentials_[i];
    if (std::find(pot.scope().begin(), pot.scope().end(), node) !=
        pot.scope().end()) {
      std::vector<NodeId> marg_vars;
      for (NodeId scope_node : pot.scope())
        if (scope_node != node)
          marg_vars.push_back(scope_node);
      return pot.marginalize(marg_vars).tensor();
    }
  }
  throw std::invalid_argument("Node not found in any calibrated clique.");
}

} // namespace bncore
