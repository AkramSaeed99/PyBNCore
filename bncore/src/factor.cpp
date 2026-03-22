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

void Factor::bind_data(double *ptr) { tensor_.bind_data(ptr); }

Factor Factor::multiply(const Factor &other, BumpAllocator *allocator) const {
  std::vector<NodeId> new_scope = scope_;
  std::vector<std::size_t> new_sizes;
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
    new_sizes.push_back(batch_size);
  }

  std::size_t num_elements = std::accumulate(new_sizes.begin(), new_sizes.end(),
                                             1ULL, std::multiplies<>());
  Factor result = allocator ? Factor(new_scope, new_sizes,
                                     allocator->allocate(num_elements))
                            : Factor(new_scope, new_sizes);

  std::size_t num_states_C = num_elements / batch_size;

  // Precompute strides for C, A, B
  const std::size_t c_ndim = new_scope.size();
  const std::size_t a_ndim = scope_.size();
  const std::size_t b_ndim = other.scope_.size();

  std::vector<std::size_t> c_strides(c_ndim);
  if (c_ndim > 0) {
    c_strides[c_ndim - 1] = 1;
    for (int i = static_cast<int>(c_ndim) - 2; i >= 0; --i)
      c_strides[i] = c_strides[i + 1] * new_sizes[i + 1];
  }

  std::vector<std::size_t> a_strides(a_ndim);
  if (a_ndim > 0) {
    a_strides[a_ndim - 1] = 1;
    for (int i = static_cast<int>(a_ndim) - 2; i >= 0; --i)
      a_strides[i] = a_strides[i + 1] * tensor_.shape()[i + 1];
  }

  std::vector<std::size_t> b_strides(b_ndim);
  if (b_ndim > 0) {
    b_strides[b_ndim - 1] = 1;
    for (int i = static_cast<int>(b_ndim) - 2; i >= 0; --i)
      b_strides[i] = b_strides[i + 1] * other.tensor_.shape()[i + 1];
  }

  // Precompute index mapping tables: for each c_idx → a_idx, b_idx
  std::vector<std::size_t> map_a(num_states_C);
  std::vector<std::size_t> map_b(num_states_C);

  for (std::size_t c_idx = 0; c_idx < num_states_C; ++c_idx) {
    // Unravel c_idx using precomputed strides
    std::size_t a_idx = 0;
    std::size_t b_idx = 0;
    std::size_t remainder = c_idx;

    for (std::size_t d = 0; d < c_ndim; ++d) {
      std::size_t coord = remainder / c_strides[d];
      remainder -= coord * c_strides[d];

      // A is prefix: dims 0..a_ndim-1 map directly
      if (d < a_ndim) {
        a_idx += coord * a_strides[d];
      }

      // B maps via b_in_c
      for (std::size_t j = 0; j < b_ndim; ++j) {
        if (b_in_c[j] == static_cast<int>(d)) {
          b_idx += coord * b_strides[j];
          break;
        }
      }
    }
    map_a[c_idx] = a_idx;
    map_b[c_idx] = b_idx;
  }

  // Execute multiply using precomputed maps
  double *c_data = result.tensor().data();
  const double *a_data = tensor_.data();
  const double *b_data = other.tensor().data();

  if (batch_size == 1) {
    // Scalar fast path — no batch loop overhead
    for (std::size_t c_idx = 0; c_idx < num_states_C; ++c_idx) {
      double val_a = a_data[map_a[c_idx]];
      double val_b = b_data[map_b[c_idx]];
      c_data[c_idx] = val_a * val_b;
    }
  } else {
    for (std::size_t c_idx = 0; c_idx < num_states_C; ++c_idx) {
      double *c_ptr = c_data + c_idx * batch_size;
      const double *a_ptr =
          a_data + map_a[c_idx] * (a_batched ? batch_size : 1);
      const double *b_ptr =
          b_data + map_b[c_idx] * (b_batched ? batch_size : 1);

#pragma omp simd
      for (std::size_t b = 0; b < batch_size; ++b) {
        double val_a = a_batched ? a_ptr[b] : a_ptr[0];
        double val_b = b_batched ? b_ptr[b] : b_ptr[0];
        c_ptr[b] = val_a * val_b;
      }
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

  // Precompute strides for A and C
  const std::size_t a_ndim = scope_.size();
  const std::size_t c_ndim = new_scope.size();

  std::vector<std::size_t> a_strides(a_ndim);
  if (a_ndim > 0) {
    a_strides[a_ndim - 1] = 1;
    for (int i = static_cast<int>(a_ndim) - 2; i >= 0; --i)
      a_strides[i] = a_strides[i + 1] * tensor_.shape()[i + 1];
  }

  std::vector<std::size_t> c_strides(c_ndim);
  if (c_ndim > 0) {
    c_strides[c_ndim - 1] = 1;
    for (int i = static_cast<int>(c_ndim) - 2; i >= 0; --i)
      c_strides[i] = c_strides[i + 1] * new_sizes[i + 1];
  }

  // Precompute a_idx → c_idx mapping
  std::vector<std::size_t> map_c(num_states_A);
  for (std::size_t a_idx = 0; a_idx < num_states_A; ++a_idx) {
    std::size_t c_idx = 0;
    std::size_t remainder = a_idx;

    for (std::size_t d = 0; d < a_ndim; ++d) {
      std::size_t coord = remainder / a_strides[d];
      remainder -= coord * a_strides[d];

      // Map only retained dimensions
      for (std::size_t k = 0; k < c_ndim; ++k) {
        if (c_in_a[k] == static_cast<int>(d)) {
          c_idx += coord * c_strides[k];
          break;
        }
      }
    }
    map_c[a_idx] = c_idx;
  }

  // Execute marginalize using precomputed maps
  double *c_data = result.tensor().data();
  const double *a_data = tensor_.data();

  if (batch_size == 1) {
    // Scalar fast path
    for (std::size_t a_idx = 0; a_idx < num_states_A; ++a_idx) {
      c_data[map_c[a_idx]] += a_data[a_idx];
    }
  } else {
    for (std::size_t a_idx = 0; a_idx < num_states_A; ++a_idx) {
      double *c_ptr = c_data + map_c[a_idx] * batch_size;
      const double *a_ptr = a_data + a_idx * batch_size;

#pragma omp simd
      for (std::size_t b = 0; b < batch_size; ++b) {
        c_ptr[b] += a_ptr[b];
      }
    }
  }
  return result;
}

} // namespace bncore
