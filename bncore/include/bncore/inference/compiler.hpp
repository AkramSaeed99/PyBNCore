#pragma once
#include "bncore/graph/graph.hpp"
#include "bncore/inference/junction_tree.hpp"
#include <memory>
#include <string>
#include <vector>

namespace bncore {

class JunctionTreeCompiler {
public:
  static std::unique_ptr<JunctionTree>
  compile(const Graph &graph, const std::string &heuristic = "min_fill");

private:
  static std::vector<std::vector<NodeId>> moralize(const Graph &graph);
};

} // namespace bncore
