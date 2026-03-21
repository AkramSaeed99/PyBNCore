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

} // namespace bncore
