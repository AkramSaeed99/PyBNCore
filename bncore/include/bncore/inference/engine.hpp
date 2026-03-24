#pragma once
#include "bncore/inference/junction_tree.hpp"
#include "bncore/inference/workspace.hpp"
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
#include "bncore/graph/node.hpp"

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

  // MAP / MPE query:
  // output_states layout is [batch_size x num_model_vars] (row-major).
  // num_vars is the evidence matrix width.
  void evaluate_map(const int *evidence_data, std::size_t batch_size,
                    std::size_t num_vars, int *output_states);

  void set_soft_evidence(bncore::NodeId var, const double *likelihoods,
                         std::size_t n_states);
  void set_soft_evidence_matrix(bncore::NodeId var,
                                const double *likelihoods_matrix,
                                std::size_t total_n_states); // size is batch_size * n_states
  void clear_soft_evidence();

  void invalidate_workspace_cache();

  const JunctionTree &junction_tree() const { return jt_; }

private:
  void apply_soft_evidence_to_workspace(BatchWorkspace &workspace,
                                        std::size_t current_batch_start,
                                        std::size_t current_chunk_size) const;

  const JunctionTree &jt_;
  std::size_t num_threads_;
  std::size_t chunk_size_;
  std::unique_ptr<BatchWorkspace> single_workspace_cache_;
  std::size_t single_workspace_batch_size_ = 0;

  std::unordered_map<bncore::NodeId, std::vector<double>> soft_evidence_scalar_;
  // mapping NodeId -> matrix [row_major, batch_size * n_states]
  std::unordered_map<bncore::NodeId, std::vector<double>> soft_evidence_matrix_;
};

} // namespace bncore
