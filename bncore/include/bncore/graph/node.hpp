#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace bncore {

using NodeId = std::uint32_t;

struct VariableMetadata {
    NodeId id;
    std::string name;
    std::vector<std::string> states;

    [[nodiscard]] std::size_t num_states() const {
        return states.size();
    }
};

} // namespace bncore
