#include "bncore/factors/factor.hpp"
#include "bncore/util/bump_allocator.hpp"

namespace bncore {

Factor::Factor(const std::vector<NodeId> &scope,
               const std::vector<std::size_t> &state_sizes)
    : scope_(scope), tensor_(state_sizes) {
  if (scope.size() != state_sizes.size() &&
      scope.size() + 1 != state_sizes.size())
    throw std::invalid_argument("state_sizes must either match scope size or "
                                "include a batch dimension.");
}

Factor::Factor(const std::vector<NodeId> &scope,
               const std::vector<std::size_t> &state_sizes,
               double *preallocated_ptr)
    : scope_(scope), tensor_(state_sizes, preallocated_ptr) {
  if (scope.size() != state_sizes.size() &&
      scope.size() + 1 != state_sizes.size())
    throw std::invalid_argument("state_sizes must either match scope size or "
                                "include a batch dimension.");
}

Factor Factor::multiply(const Factor &other, BumpAllocator *allocator) const {
  std::vector<NodeId> new_scope = scope_;
  std::vector<std::size_t> new_sizes;
  // Get out own sizes excluding batch
  for (std::size_t i = 0; i < scope_.size(); ++i)
    new_sizes.push_back(tensor_.shape()[i]);

  std::vector<int> b_in_c(other.scope_.size(), -1);
  for (std::size_t j = 0; j < other.scope_.size(); ++j) {
    auto it = std::find(new_scope.begin(), new_scope.end(), other.scope_[j]);
    if (it == new_scope.end()) {
      new_scope.push_back(other.scope_[j]);
      new_sizes.push_back(other.tensor_.shape()[j]);
      b_in_c[j] = new_scope.size() - 1;
    } else {
      b_in_c[j] = std::distance(new_scope.begin(), it);
    }
  }

  bool a_batched = (scope_.size() + 1 == tensor_.shape().size());
  bool b_batched = (other.scope_.size() + 1 == other.tensor_.shape().size());
  std::size_t batch_size = a_batched
                               ? tensor_.shape().back()
                               : (b_batched ? other.tensor_.shape().back() : 1);

  if (a_batched || b_batched) {
    new_sizes.push_back(batch_size); // Ensure innermost is batch dim
  }

  std::size_t num_elements = std::accumulate(new_sizes.begin(), new_sizes.end(),
                                             1ULL, std::multiplies<>());
  Factor result = allocator ? Factor(new_scope, new_sizes,
                                     allocator->allocate(num_elements))
                            : Factor(new_scope, new_sizes);

  std::size_t num_states_C = num_elements / batch_size;
  std::size_t num_states_A = tensor_.size() / (a_batched ? batch_size : 1);
  std::size_t num_states_B =
      other.tensor().size() / (b_batched ? batch_size : 1);

  std::vector<std::size_t> c_indices(new_scope.size());
  std::vector<std::size_t> a_indices(scope_.size());
  std::vector<std::size_t> b_indices(other.scope_.size());

  // To avoid writing a fully recursive unraveller, we use a simple flat map
  // here for v1 MVP Ideally, precompute strides into a linear mapping table
  for (std::size_t c_idx = 0; c_idx < num_states_C; ++c_idx) {
    // Unravel C
    std::size_t temp = c_idx;
    std::size_t stride = 1;
    for (int i = static_cast<int>(new_scope.size()) - 1; i >= 0; --i) {
      c_indices[i] = (temp / stride) % new_sizes[i];
      stride *= new_sizes[i];
    }

    // Map to A
    for (std::size_t i = 0; i < scope_.size(); ++i)
      a_indices[i] = c_indices[i]; // A is prefix

    // Map to B
    for (std::size_t j = 0; j < other.scope_.size(); ++j)
      b_indices[j] = c_indices[b_in_c[j]];

    // Ravel A and B
    std::size_t a_idx = 0;
    stride = 1;
    for (int i = static_cast<int>(scope_.size()) - 1; i >= 0; --i) {
      a_idx += a_indices[i] * stride;
      stride *= tensor_.shape()[i];
    }
    std::size_t b_idx = 0;
    stride = 1;
    for (int i = static_cast<int>(other.scope_.size()) - 1; i >= 0; --i) {
      b_idx += b_indices[i] * stride;
      stride *= other.tensor_.shape()[i];
    }

    double *c_ptr = result.tensor().data() + c_idx * batch_size;
    const double *a_ptr = tensor_.data() + a_idx * (a_batched ? batch_size : 1);
    const double *b_ptr =
        other.tensor().data() + b_idx * (b_batched ? batch_size : 1);

#pragma omp simd
    for (std::size_t b = 0; b < batch_size; ++b) {
      double val_a = a_batched ? a_ptr[b] : a_ptr[0];
      double val_b = b_batched ? b_ptr[b] : b_ptr[0];
      c_ptr[b] = val_a * val_b;
    }
  }
  return result;
}

Factor Factor::marginalize(const std::vector<NodeId> &marg_vars,
                           BumpAllocator *allocator) const {
  std::vector<NodeId> new_scope;
  std::vector<std::size_t> new_sizes;
  std::vector<int> c_in_a;
  for (std::size_t i = 0; i < scope_.size(); ++i) {
    if (std::find(marg_vars.begin(), marg_vars.end(), scope_[i]) ==
        marg_vars.end()) {
      new_scope.push_back(scope_[i]);
      new_sizes.push_back(tensor_.shape()[i]);
      c_in_a.push_back(i);
    }
  }

  bool is_batched = (scope_.size() + 1 == tensor_.shape().size());
  std::size_t batch_size = is_batched ? tensor_.shape().back() : 1;

  if (is_batched) {
    new_sizes.push_back(batch_size);
  }

  std::size_t num_elements = std::accumulate(new_sizes.begin(), new_sizes.end(),
                                             1ULL, std::multiplies<>());
  Factor result = allocator ? Factor(new_scope, new_sizes,
                                     allocator->allocate(num_elements))
                            : Factor(new_scope, new_sizes);

  if (allocator) {
    result.tensor().fill(0.0);
  }

  std::size_t num_states_A = tensor_.size() / batch_size;

  std::vector<std::size_t> a_indices(scope_.size());

  for (std::size_t a_idx = 0; a_idx < num_states_A; ++a_idx) {
    std::size_t temp = a_idx;
    std::size_t stride = 1;
    for (int i = static_cast<int>(scope_.size()) - 1; i >= 0; --i) {
      a_indices[i] = (temp / stride) % tensor_.shape()[i];
      stride *= tensor_.shape()[i];
    }

    std::size_t c_idx = 0;
    stride = 1;
    for (int i = static_cast<int>(new_scope.size()) - 1; i >= 0; --i) {
      c_idx += a_indices[c_in_a[i]] * stride;
      stride *= new_sizes[i];
    }

    double *c_ptr = result.tensor().data() + c_idx * batch_size;
    const double *a_ptr = tensor_.data() + a_idx * batch_size;

#pragma omp simd
    for (std::size_t b = 0; b < batch_size; ++b) {
      c_ptr[b] += a_ptr[b];
    }
  }
  return result;
}

} // namespace bncore
