#pragma once
#include "bncore/factors/dense_tensor.hpp"
#include "bncore/graph/node.hpp"
#include <vector>

namespace bncore {

class Factor {
public:
  Factor(const std::vector<NodeId> &scope,
         const std::vector<std::size_t> &state_sizes);
  Factor(const std::vector<NodeId> &scope,
         const std::vector<std::size_t> &state_sizes, double *preallocated_ptr);

  [[nodiscard]] const std::vector<NodeId> &scope() const { return scope_; }

  [[nodiscard]] Factor multiply(const Factor &other,
                                class BumpAllocator *allocator = nullptr) const;
  [[nodiscard]] Factor
  marginalize(const std::vector<NodeId> &marg_vars,
              class BumpAllocator *allocator = nullptr) const;

  DenseTensor &tensor() { return tensor_; }
  [[nodiscard]] const DenseTensor &tensor() const { return tensor_; }

  void bind_data(double *ptr);

  // Core math operations

private:
  std::vector<NodeId> scope_;
  DenseTensor tensor_;
};

} // namespace bncore
