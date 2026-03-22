#pragma once
#include "bncore/inference/junction_tree.hpp"
#include "bncore/inference/workspace.hpp"
#include <cstdint>
#include <future>
#include <memory>
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

  void evaluate_multi(const int *evidence_data, std::size_t batch_size,
                      std::size_t num_vars, const std::size_t *query_vars,
                      std::size_t num_queries,
                      const std::size_t *output_offsets,
                      double *output_data);

  void invalidate_workspace_cache();

  const JunctionTree &junction_tree() const { return jt_; }

private:
  const JunctionTree &jt_;
  std::size_t num_threads_;
  std::size_t chunk_size_;
  std::unique_ptr<BatchWorkspace> single_workspace_cache_;
  std::size_t single_workspace_batch_size_ = 0;
};

} // namespace bncore
