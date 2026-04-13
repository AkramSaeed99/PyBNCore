#include "bncore/inference/junction_tree.hpp"

namespace bncore {

JunctionTree::JunctionTree(const Graph *graph) : graph_(graph) {}

void JunctionTree::add_clique(const std::vector<NodeId> &scope,
                              const std::vector<std::size_t> &state_sizes) {
  cliques_.push_back(
      {cliques_.size(), scope, {}, Factor(scope, state_sizes), {}});
}

void JunctionTree::add_separator(std::size_t c1, std::size_t c2,
                                 const std::vector<NodeId> &scope) {
  separators_.push_back({c1, c2, scope});
  cliques_[c1].neighbor_cliques.push_back(c2);
  cliques_[c2].neighbor_cliques.push_back(c1);
}

JunctionTree::Stats JunctionTree::stats() const {
  Stats s{};
  s.num_cliques = cliques_.size();
  s.max_clique_size = 0;
  s.total_table_entries = 0;

  for (const auto &clq : cliques_) {
    std::size_t scope_size = clq.scope.size();
    if (scope_size > s.max_clique_size) s.max_clique_size = scope_size;

    std::size_t entries = 1;
    for (NodeId nid : clq.scope)
      entries *= graph_->get_variable(nid).states.size();
    s.total_table_entries += entries;
  }

  s.treewidth = s.max_clique_size > 0 ? s.max_clique_size - 1 : 0;
  return s;
}

} // namespace bncore
