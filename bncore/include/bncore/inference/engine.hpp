#pragma once
#include "bncore/inference/junction_tree.hpp"
#include "bncore/inference/workspace.hpp"
#include "bncore/util/thread_pool.hpp"
#include <cstdint>
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

  // Runtime toggle for d-separation pruning (default: enabled).
  // Applies to all workspaces created by this engine. Primarily for
  // benchmarking the impact of Bayes-Ball pruning.
  void set_dsep_enabled(bool enabled);
  bool dsep_enabled() const { return dsep_enabled_; }

  const JunctionTree &junction_tree() const { return jt_; }

private:
  void apply_soft_evidence_to_workspace(BatchWorkspace &workspace,
                                        std::size_t current_batch_start,
                                        std::size_t current_chunk_size) const;

  // Get or create a workspace from the pool for the given thread index.
  BatchWorkspace &get_pooled_workspace(std::size_t thread_idx);

  const JunctionTree &jt_;
  std::size_t num_threads_;
  std::size_t chunk_size_;
  std::unique_ptr<BatchWorkspace> single_workspace_cache_;
  std::size_t single_workspace_batch_size_ = 0;
  bool dsep_enabled_ = true;

  // Persistent thread pool — eliminates per-call thread creation overhead
  std::unique_ptr<ThreadPool> thread_pool_;

  // Workspace pool — one per thread, reused across evaluate() calls
  std::vector<std::unique_ptr<BatchWorkspace>> workspace_pool_;

  std::unordered_map<bncore::NodeId, std::vector<double>> soft_evidence_scalar_;
  // mapping NodeId -> matrix [row_major, batch_size * n_states]
  std::unordered_map<bncore::NodeId, std::vector<double>> soft_evidence_matrix_;
};

} // namespace bncore
