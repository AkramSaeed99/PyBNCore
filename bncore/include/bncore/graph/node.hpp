#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace bncore {

using NodeId = std::uint32_t;

struct VariableMetadata {
  NodeId id;
  std::string name;
  std::vector<std::string> states;
  std::vector<double> cpt;

  [[nodiscard]] std::size_t num_states() const { return states.size(); }
};

} // namespace bncore
