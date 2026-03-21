#pragma once
#include "bncore/inference/junction_tree.hpp"
#include "bncore/inference/workspace.hpp"
#include <cstdint>
#include <future>
#include <thread>
#include <vector>

namespace bncore {

class BatchExecutionEngine {
public:
  BatchExecutionEngine(const JunctionTree &jt, std::size_t num_threads,
                       std::size_t chunk_size);

  void evaluate(const int *evidence_data, std::size_t batch_size,
                std::size_t num_vars, double *output_data,
                std::size_t query_var);

private:
  const JunctionTree &jt_;
  std::size_t num_threads_;
  std::size_t chunk_size_;
};

} // namespace bncore
