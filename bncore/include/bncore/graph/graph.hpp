#pragma once
#include "bncore/graph/node.hpp"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace bncore {

class Graph {
public:
  Graph() = default;

  NodeId add_variable(const std::string &name,
                      const std::vector<std::string> &states);
  void split_state(NodeId id, std::size_t state_idx,
                   const std::string &new_state1,
                   const std::string &new_state2);
  void add_edge(NodeId parent, NodeId child);
  void add_edge(const std::string &parent_name, const std::string &child_name);

  void set_cpt(NodeId id, const std::vector<double> &cpt);
  void set_cpt(const std::string &name, const std::vector<double> &cpt);

  // Validate all CPTs: check [0,1] bounds, sum-to-1 per conditional row,
  // and no NaN/Inf values.  Throws std::invalid_argument on first violation.
  // tolerance is the allowed deviation from 1.0 for row sums.
  void validate_cpts(double tolerance = 1e-6) const;

  [[nodiscard]] const VariableMetadata &get_variable(NodeId id) const;
  [[nodiscard]] const VariableMetadata &
  get_variable(const std::string &name) const;

  [[nodiscard]] const std::vector<NodeId> &get_parents(NodeId id) const;
  [[nodiscard]] const std::vector<NodeId> &get_children(NodeId id) const;

  [[nodiscard]] std::size_t num_variables() const;

private:
  std::vector<VariableMetadata> variables_;
  std::unordered_map<std::string, NodeId> name_to_id_;

  // Adjacency lists
  std::vector<std::vector<NodeId>> parents_;
  std::vector<std::vector<NodeId>> children_;
};

} // namespace bncore
