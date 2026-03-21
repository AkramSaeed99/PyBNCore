#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bncore {

/**
 * @brief Thread-local zero-allocation scratchpad for contiguous inference
 * arrays. Bumps an internal pointer for O(1) lightning-fast allocations during
 * inner message passing. Operates across the vectorized batch chunks entirely
 * lock-free.
 */
class BumpAllocator {
public:
  explicit BumpAllocator(std::size_t capacity_doubles)
      : buffer_(capacity_doubles, 0.0), offset_(0) {}

  [[nodiscard]] double *allocate(std::size_t num_doubles) {
    if (offset_ + num_doubles > buffer_.size()) {
      throw std::bad_alloc();
    }
    double *ptr = &buffer_[offset_];
    offset_ += num_doubles;
    return ptr;
  }

  // O(1) memory cleanup when moving to the next batch chunk.
  void reset() { offset_ = 0; }

  [[nodiscard]] std::size_t capacity() const { return buffer_.size(); }
  [[nodiscard]] std::size_t used() const { return offset_; }

private:
  std::vector<double> buffer_;
  std::size_t offset_;
};

} // namespace bncore
