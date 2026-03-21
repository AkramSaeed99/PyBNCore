#pragma once
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace bncore {

class DenseTensor {
public:
  explicit DenseTensor(const std::vector<std::size_t> &shape);
  // Borrows memory avoiding system allocation locks
  DenseTensor(const std::vector<std::size_t> &shape, double *preallocated_ptr);

  [[nodiscard]] const std::vector<std::size_t> &shape() const;
  [[nodiscard]] std::size_t size() const;

  [[nodiscard]] double &operator[](std::size_t flat_idx);
  [[nodiscard]] double operator[](std::size_t flat_idx) const;

  [[nodiscard]] double &at(const std::vector<std::size_t> &indices);
  [[nodiscard]] double at(const std::vector<std::size_t> &indices) const;

  double *data();
  const double *data() const;

  void multiply_inplace(const DenseTensor &other);
  void fill(double value);

private:
  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;

  std::vector<double> owned_data_;
  double *mapped_data_ = nullptr;
  std::size_t num_elements_;

  void compute_strides();
  [[nodiscard]] std::size_t
  compute_flat_index(const std::vector<std::size_t> &indices) const;
};

} // namespace bncore
