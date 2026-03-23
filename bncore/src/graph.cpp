#include "bncore/graph/graph.hpp"

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

void Graph::split_state(NodeId id, std::size_t state_idx,
                        const std::string &new_state1,
                        const std::string &new_state2) {
  if (id >= variables_.size())
    throw std::out_of_range("Invalid NodeId.");
  auto &var = variables_[id];
  if (state_idx >= var.states.size())
    throw std::out_of_range("Invalid state index.");

  var.states.erase(var.states.begin() + state_idx);
  var.states.insert(var.states.begin() + state_idx, new_state2);
  var.states.insert(var.states.begin() + state_idx, new_state1);
}

void Graph::add_edge(NodeId parent, NodeId child) {
  if (parent >= variables_.size() || child >= variables_.size()) {
    throw std::out_of_range("Invalid NodeId provided for edge.");
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

} // namespace bncore
