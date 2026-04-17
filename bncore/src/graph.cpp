#include "bncore/graph/graph.hpp"
#include <cmath>

namespace bncore {

NodeId Graph::add_variable(const std::string &name,
                           const std::vector<std::string> &states) {
  if (name_to_id_.contains(name)) {
    throw std::invalid_argument("Variable with name '" + name +
                                "' already exists.");
  }

  NodeId id = static_cast<NodeId>(variables_.size());
  variables_.push_back({id, name, states});
  name_to_id_[name] = id;

  parents_.emplace_back();
  children_.emplace_back();

  return id;
}

// Helper: duplicate the slice at (axis = split_axis, index = slot) in a flat
// tensor whose logical shape is outer × axis_size × inner.  Returns a new
// flat vector whose shape is outer × (axis_size+1) × inner, with the slot's
// values present at BOTH slot and slot+1.  Caller may rescale afterward.
static std::vector<double> insert_duplicate_slice(
    const std::vector<double> &data,
    std::size_t outer, std::size_t axis_size, std::size_t inner,
    std::size_t slot) {
  const std::size_t new_axis = axis_size + 1;
  std::vector<double> out(outer * new_axis * inner, 0.0);
  for (std::size_t o = 0; o < outer; ++o) {
    for (std::size_t a = 0; a < axis_size; ++a) {
      const std::size_t a_new = (a <= slot) ? a : a + 1;
      for (std::size_t i = 0; i < inner; ++i) {
        const double v = data[(o * axis_size + a) * inner + i];
        out[(o * new_axis + a_new) * inner + i] = v;
        if (a == slot) {
          // duplicate into slot+1
          out[(o * new_axis + slot + 1) * inner + i] = v;
        }
      }
    }
  }
  return out;
}

void Graph::split_state(NodeId id, std::size_t state_idx,
                        const std::string &new_state1,
                        const std::string &new_state2) {
  if (id >= variables_.size())
    throw std::out_of_range("Invalid NodeId.");
  auto &var = variables_[id];
  if (state_idx >= var.states.size())
    throw std::out_of_range("Invalid state index.");

  const std::size_t old_k = var.states.size();

  // ── Step 1: fix X's OWN CPT (if present) ────────────────────────────────
  // CPT layout: flat [pa_configs, k_X, (B)] with child axis last (before batch).
  // When X has no CPT yet (e.g. before set_cpt was called), skip.
  if (!var.cpt.empty()) {
    std::size_t pa_configs = 1;
    for (NodeId p : parents_[id]) pa_configs *= variables_[p].states.size();
    const std::size_t family_states = pa_configs * old_k;
    if (var.cpt.size() % family_states != 0) {
      throw std::invalid_argument(
          "split_state: CPT size of variable '" + var.name +
          "' is inconsistent with its state count.");
    }
    const std::size_t batch = var.cpt.size() / family_states;

    // outer = pa_configs, axis = old_k, inner = batch
    auto grown = insert_duplicate_slice(var.cpt, pa_configs, old_k, batch,
                                        state_idx);
    // Mass conservation: the two new states must replace one old state so
    // that each parent-conditioned row still sums to 1.  Halve both copies.
    const std::size_t new_k = old_k + 1;
    for (std::size_t o = 0; o < pa_configs; ++o) {
      for (std::size_t b = 0; b < batch; ++b) {
        std::size_t il = (o * new_k + state_idx) * batch + b;
        std::size_t ir = (o * new_k + state_idx + 1) * batch + b;
        grown[il] *= 0.5;
        grown[ir] *= 0.5;
      }
    }
    var.cpt = std::move(grown);
  }

  // ── Step 2: fix every CHILD's CPT where X is a parent ───────────────────
  // Child Y's CPT shape: [pa_of_Y..., k_Y, (B)].  X is at position pos_X
  // among Y's parents (the order in parents_[Y]).  We compute:
  //   outer = product of cardinalities of parents BEFORE X in Y's parent list
  //   axis  = old_k (X's old cardinality)
  //   inner = (product of cardinalities of parents AFTER X) * k_Y * batch
  // then duplicate X's slot.  Values are not halved — children's conditional
  // distributions at the split state remain valid for BOTH new states.
  for (NodeId child_id : children_[id]) {
    auto &child = variables_[child_id];
    if (child.cpt.empty()) continue;
    const std::size_t k_Y = child.states.size();

    // Find X's position in Y's parents.
    const auto &pa = parents_[child_id];
    std::size_t pos_X = 0;
    bool found = false;
    for (std::size_t i = 0; i < pa.size(); ++i) {
      if (pa[i] == id) { pos_X = i; found = true; break; }
    }
    if (!found) continue;  // should not happen

    std::size_t outer = 1;
    for (std::size_t i = 0; i < pos_X; ++i)
      outer *= variables_[pa[i]].states.size();
    std::size_t tail = 1;
    for (std::size_t i = pos_X + 1; i < pa.size(); ++i)
      tail *= variables_[pa[i]].states.size();
    const std::size_t total_pa_configs = outer * old_k * tail;
    const std::size_t family_states = total_pa_configs * k_Y;
    if (child.cpt.size() % family_states != 0) {
      throw std::invalid_argument(
          "split_state: CPT size of child '" + child.name +
          "' is inconsistent with its family state count.");
    }
    const std::size_t batch = child.cpt.size() / family_states;
    const std::size_t inner = tail * k_Y * batch;

    child.cpt = insert_duplicate_slice(child.cpt, outer, old_k, inner,
                                        state_idx);
  }

  // ── Step 3: update X's state list ───────────────────────────────────────
  var.states.erase(var.states.begin() + state_idx);
  var.states.insert(var.states.begin() + state_idx, new_state2);
  var.states.insert(var.states.begin() + state_idx, new_state1);
}

void Graph::add_edge(NodeId parent, NodeId child) {
  if (parent >= variables_.size() || child >= variables_.size()) {
    throw std::out_of_range("Invalid NodeId provided for edge.");
  }
  if (parent == child) {
    throw std::invalid_argument(
        "Self-loop detected: cannot add edge from '" +
        variables_[parent].name + "' to itself.");
  }

  // Cycle detection: check if child can reach parent via existing edges.
  // If so, adding parent→child would create a cycle.
  {
    std::vector<bool> visited(variables_.size(), false);
    std::vector<NodeId> stack;
    stack.push_back(parent);
    visited[parent] = true;
    while (!stack.empty()) {
      NodeId u = stack.back();
      stack.pop_back();
      for (NodeId p : parents_[u]) {
        if (p == child) {
          throw std::invalid_argument(
              "Cycle detected: adding edge '" + variables_[parent].name +
              "' -> '" + variables_[child].name +
              "' would create a cycle in the DAG.");
        }
        if (!visited[p]) {
          visited[p] = true;
          stack.push_back(p);
        }
      }
    }
  }

  parents_[child].push_back(parent);
  children_[parent].push_back(child);
}

void Graph::add_edge(const std::string &parent_name,
                     const std::string &child_name) {
  add_edge(get_variable(parent_name).id, get_variable(child_name).id);
}

void Graph::set_cpt(NodeId id, const std::vector<double> &cpt) {
  if (id >= variables_.size())
    throw std::out_of_range("Invalid NodeId");

  std::size_t family_states = variables_[id].states.size();
  for (NodeId p : parents_[id]) {
    family_states *= variables_[p].states.size();
  }

  if (cpt.empty()) {
    throw std::invalid_argument("CPT for variable '" + variables_[id].name +
                                "' cannot be empty. Expected scalar size " +
                                std::to_string(family_states) +
                                " or batched size " +
                                std::to_string(family_states) + " * B.");
  }

  if (cpt.size() == family_states) {
    variables_[id].cpt = cpt;
    return;
  }

  if (cpt.size() % family_states != 0) {
    throw std::invalid_argument(
        "CPT size mismatch for variable '" + variables_[id].name +
        "'. Expected scalar size " + std::to_string(family_states) +
        " or batched size " + std::to_string(family_states) +
        " * B (B > 0), but got " + std::to_string(cpt.size()) + ".");
  }

  std::size_t batch_size = cpt.size() / family_states;
  if (batch_size == 0) {
    throw std::invalid_argument("CPT for variable '" + variables_[id].name +
                                "' has invalid batch size 0.");
  }

  variables_[id].cpt = cpt;
}

void Graph::set_cpt(const std::string &name, const std::vector<double> &cpt) {
  set_cpt(name_to_id_.at(name), cpt);
}

const VariableMetadata &Graph::get_variable(NodeId id) const {
  if (id >= variables_.size())
    throw std::out_of_range("Invalid NodeId.");
  return variables_[id];
}

const VariableMetadata &Graph::get_variable(const std::string &name) const {
  auto it = name_to_id_.find(name);
  if (it == name_to_id_.end())
    throw std::invalid_argument("Variable not found.");
  return variables_[it->second];
}

const std::vector<NodeId> &Graph::get_parents(NodeId id) const {
  if (id >= parents_.size())
    throw std::out_of_range("Invalid NodeId.");
  return parents_[id];
}

const std::vector<NodeId> &Graph::get_children(NodeId id) const {
  if (id >= children_.size())
    throw std::out_of_range("Invalid NodeId.");
  return children_[id];
}

std::size_t Graph::num_variables() const { return variables_.size(); }

void Graph::validate_cpts(double tolerance) const {
  for (const auto &var : variables_) {
    if (var.cpt.empty()) {
      throw std::invalid_argument(
          "validate_cpts: variable '" + var.name + "' has no CPT assigned.");
    }

    const std::size_t n_states = var.states.size();
    std::size_t family_states = n_states;
    for (NodeId p : parents_[var.id]) {
      family_states *= variables_[p].states.size();
    }

    const std::size_t batch_size = var.cpt.size() / family_states;
    const std::size_t n_parent_configs = family_states / n_states;

    for (std::size_t b = 0; b < batch_size; ++b) {
      for (std::size_t pc = 0; pc < n_parent_configs; ++pc) {
        double row_sum = 0.0;
        for (std::size_t s = 0; s < n_states; ++s) {
          // CPT layout: [parent_config * n_states + state] for scalar,
          // or [entry * batch_size + b] for batched.
          std::size_t idx;
          if (batch_size == 1) {
            idx = pc * n_states + s;
          } else {
            idx = (pc * n_states + s) * batch_size + b;
          }

          double val = var.cpt[idx];

          if (std::isnan(val) || std::isinf(val)) {
            throw std::invalid_argument(
                "validate_cpts: variable '" + var.name +
                "' has NaN/Inf in CPT at index " + std::to_string(idx) + ".");
          }
          if (val < 0.0 || val > 1.0) {
            throw std::invalid_argument(
                "validate_cpts: variable '" + var.name +
                "' has probability " + std::to_string(val) +
                " outside [0,1] at index " + std::to_string(idx) + ".");
          }
          row_sum += val;
        }

        if (std::abs(row_sum - 1.0) > tolerance) {
          throw std::invalid_argument(
              "validate_cpts: variable '" + var.name +
              "' conditional distribution sums to " +
              std::to_string(row_sum) + " (expected 1.0, tolerance " +
              std::to_string(tolerance) + ") for parent config " +
              std::to_string(pc) + ", batch " + std::to_string(b) + ".");
        }
      }
    }
  }
}

} // namespace bncore
