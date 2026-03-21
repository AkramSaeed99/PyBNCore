#pragma once
#include "bncore/factors/factor.hpp"
#include "bncore/graph/graph.hpp"
#include <vector>

namespace bncore {

struct Clique {
  std::size_t id;
  std::vector<NodeId> scope;
  std::vector<NodeId> assigned_cpts;
  Factor base_potential; // Uncalibrated potential
  std::vector<std::size_t> neighbor_cliques;
};

struct Separator {
  std::size_t clique1;
  std::size_t clique2;
  std::vector<NodeId> scope;
};

class JunctionTree {
public:
  explicit JunctionTree(const Graph *graph);

  void add_clique(const std::vector<NodeId> &scope,
                  const std::vector<std::size_t> &state_sizes);
  void add_separator(std::size_t c1, std::size_t c2,
                     const std::vector<NodeId> &scope);

  [[nodiscard]] const Graph *graph() const { return graph_; }
  [[nodiscard]] const std::vector<Clique> &cliques() const { return cliques_; }
  [[nodiscard]] const std::vector<Separator> &separators() const {
    return separators_;
  }

  std::vector<Clique> &get_mutable_cliques() { return cliques_; }

private:
  const Graph *graph_;
  std::vector<Clique> cliques_;
  std::vector<Separator> separators_;
};

} // namespace bncore
