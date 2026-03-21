#pragma once
#include "bncore/graph/graph.hpp"
#include <cstdint>
#include <vector>

namespace bncore {

class DiscretizationManager {
public:
  explicit DiscretizationManager(std::size_t max_bins_per_var = 50);

  bool should_split(const Graph &graph, NodeId var) const;

  void split_bin(Graph &graph, NodeId var, std::size_t state_idx);

private:
  std::size_t max_bins_per_var_;
};

} // namespace bncore
