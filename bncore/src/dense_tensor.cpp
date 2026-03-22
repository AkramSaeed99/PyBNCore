#include "bncore/factors/dense_tensor.hpp"

namespace bncore {

DenseTensor::DenseTensor(const std::vector<std::size_t> &shape)
    : shape_(shape), mapped_data_(nullptr) {
  if (shape.empty())
    throw std::invalid_argument("Tensor shape cannot be empty.");
  compute_strides();
  num_elements_ =
      std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<>());
  owned_data_.resize(num_elements_, 0.0);
}

DenseTensor::DenseTensor(const std::vector<std::size_t> &shape,
                         double *preallocated_ptr)
    : shape_(shape), mapped_data_(preallocated_ptr) {
  if (shape.empty())
    throw std::invalid_argument("Tensor shape cannot be empty.");
  if (!preallocated_ptr)
    throw std::invalid_argument("Preallocated pointer is null.");
  compute_strides();
  num_elements_ =
      std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<>());
}

const std::vector<std::size_t> &DenseTensor::shape() const { return shape_; }

std::size_t DenseTensor::size() const { return num_elements_; }

double &DenseTensor::operator[](std::size_t flat_idx) {
  return mapped_data_ ? mapped_data_[flat_idx] : owned_data_[flat_idx];
}

double DenseTensor::operator[](std::size_t flat_idx) const {
  return mapped_data_ ? mapped_data_[flat_idx] : owned_data_[flat_idx];
}

double &DenseTensor::at(const std::vector<std::size_t> &indices) {
  return this->operator[](compute_flat_index(indices));
}

double DenseTensor::at(const std::vector<std::size_t> &indices) const {
  return this->operator[](compute_flat_index(indices));
}

double *DenseTensor::data() {
  return mapped_data_ ? mapped_data_ : owned_data_.data();
}

const double *DenseTensor::data() const {
  return mapped_data_ ? mapped_data_ : owned_data_.data();
}

void DenseTensor::bind_data(double *ptr) { mapped_data_ = ptr; }

void DenseTensor::multiply_inplace(const DenseTensor &other) {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Shapes must match for inplace multiplication.");
  }
  double *my_data = data();
  const double *other_data = other.data();
  for (std::size_t i = 0; i < num_elements_; ++i) {
    my_data[i] *= other_data[i];
  }
}

void DenseTensor::fill(double value) {
  double *my_data = data();
  std::fill(my_data, my_data + num_elements_, value);
}

void DenseTensor::compute_strides() {
  strides_.resize(shape_.size());
  std::size_t stride = 1;
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape_[i];
  }
}

std::size_t
DenseTensor::compute_flat_index(const std::vector<std::size_t> &indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument(
        "Number of indices must match tensor dimensionality.");
  }
  std::size_t flat_idx = 0;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Index out of bounds for tensor dimension.");
    }
    flat_idx += indices[i] * strides_[i];
  }
  return flat_idx;
}

} // namespace bncore
