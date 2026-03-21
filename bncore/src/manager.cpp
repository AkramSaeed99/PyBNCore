#include "bncore/discretization/manager.hpp"

namespace bncore {

DiscretizationManager::DiscretizationManager(std::size_t max_bins_per_var)
    : max_bins_per_var_(max_bins_per_var) {}

bool DiscretizationManager::should_split(const Graph &graph, NodeId var) const {
  const auto &metadata = graph.get_variable(var);
  return metadata.num_states() < max_bins_per_var_;
}

void DiscretizationManager::split_bin(Graph &graph, NodeId var,
                                      std::size_t state_idx) {
  if (!should_split(graph, var))
    return;
  const auto &metadata = graph.get_variable(var);

  std::string old_name = metadata.states[state_idx];
  graph.split_state(var, state_idx, old_name + "_low", old_name + "_high");
}

} // namespace bncore
