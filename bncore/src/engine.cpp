#include "bncore/inference/engine.hpp"
#include <algorithm>
#include <stdexcept>

namespace {
void validate_commercial_license() {
  // Intentionally disabled for local/HCL integration builds.
}
} // namespace

namespace bncore {

BatchExecutionEngine::BatchExecutionEngine(const JunctionTree &jt,
                                           std::size_t num_threads,
                                           std::size_t chunk_size)
    : jt_(jt), num_threads_(num_threads),
      chunk_size_(std::max<std::size_t>(1, chunk_size)) {
  if (num_threads_ == 0)
    num_threads_ = std::thread::hardware_concurrency();
  if (num_threads_ == 0)
    num_threads_ = 1;

  // Create persistent thread pool (threads live for engine lifetime)
  if (num_threads_ > 1) {
    thread_pool_ = std::make_unique<ThreadPool>(num_threads_);
  }
}

BatchWorkspace &BatchExecutionEngine::get_pooled_workspace(std::size_t thread_idx) {
  if (workspace_pool_.size() <= thread_idx) {
    workspace_pool_.resize(thread_idx + 1);
  }
  if (!workspace_pool_[thread_idx]) {
    workspace_pool_[thread_idx] = std::make_unique<BatchWorkspace>(jt_, chunk_size_);
  }
  return *workspace_pool_[thread_idx];
}

void BatchExecutionEngine::evaluate(const int *evidence_data,
                                    std::size_t batch_size,
                                    std::size_t num_vars, double *output_data,
                                    std::size_t query_var) {
  const std::size_t num_states =
      jt_.graph()->get_variable(query_var).states.size();
  const std::size_t query_vars[1] = {query_var};
  const std::size_t offsets[2] = {0, num_states};
  evaluate_multi(evidence_data, batch_size, num_vars, query_vars, 1, offsets,
                 output_data);
}

void BatchExecutionEngine::evaluate_multi(const int *evidence_data,
                                          std::size_t batch_size,
                                          std::size_t num_vars,
                                          const std::size_t *query_vars,
                                          std::size_t num_queries,
                                          const std::size_t *output_offsets,
                                          double *output_data) {
  validate_commercial_license();
  if (!query_vars || !output_offsets || !output_data) {
    throw std::invalid_argument(
        "BatchExecutionEngine::evaluate_multi received null query/output buffers.");
  }
  if (num_queries == 0 || batch_size == 0) {
    return;
  }
  const std::size_t total_states = output_offsets[num_queries];
  if (total_states == 0) {
    return;
  }

  std::size_t num_chunks = (batch_size + chunk_size_ - 1) / chunk_size_;

  // Convert size_t query_vars to NodeId (uint32_t) once here.
  std::vector<bncore::NodeId> qvars_nodeid(num_queries);
  for (std::size_t i = 0; i < num_queries; ++i)
    qvars_nodeid[i] = static_cast<bncore::NodeId>(query_vars[i]);

  auto write_query_outputs = [&](BatchWorkspace &workspace,
                                 std::size_t current_batch_start) {
    double *out_row = output_data + current_batch_start * total_states;
    workspace.query_marginals_multi(qvars_nodeid.data(), num_queries,
                                    output_offsets, out_row);
  };

  // Fast path for single-chunk (most common Python/HCL usage)
  if (num_chunks == 1) {
    const std::size_t current_chunk_size = batch_size;
    if (!single_workspace_cache_ ||
        single_workspace_batch_size_ != current_chunk_size) {
      single_workspace_cache_ =
          std::make_unique<BatchWorkspace>(jt_, current_chunk_size);
      single_workspace_batch_size_ = current_chunk_size;
    } else {
      single_workspace_cache_->reset(current_chunk_size, 0);
    }

    BatchWorkspace &workspace = *single_workspace_cache_;
    const int *chunk_evidence = evidence_data;
    if (chunk_evidence) {
      workspace.set_evidence_matrix(chunk_evidence, num_vars);
    } else {
      workspace.clear_evidence();
    }
    apply_soft_evidence_to_workspace(workspace, 0, current_chunk_size);
    workspace.set_query_scope(qvars_nodeid.data(), num_queries);
    workspace.calibrate();
    write_query_outputs(workspace, /*current_batch_start=*/0);
    return;
  }

  // Multi-chunk processing with thread pool
  auto process_chunks = [&](std::size_t thread_idx,
                            std::size_t start_chunk,
                            std::size_t end_chunk) {
    BatchWorkspace &workspace = get_pooled_workspace(thread_idx);

    for (std::size_t c = start_chunk; c < end_chunk; ++c) {
      std::size_t current_batch_start = c * chunk_size_;
      std::size_t current_chunk_size =
          std::min(chunk_size_, batch_size - current_batch_start);

      workspace.reset(current_chunk_size, current_batch_start);

      const int *chunk_evidence =
          evidence_data ? (evidence_data + current_batch_start * num_vars)
                        : nullptr;
      if (chunk_evidence)
        workspace.set_evidence_matrix(chunk_evidence, num_vars);

      apply_soft_evidence_to_workspace(workspace, current_batch_start, current_chunk_size);
      workspace.set_query_scope(qvars_nodeid.data(), num_queries);
      workspace.calibrate();
      write_query_outputs(workspace, current_batch_start);
    }
  };

  const std::size_t effective_threads =
      std::max<std::size_t>(1, std::min(num_threads_, num_chunks));

  if (effective_threads == 1 || !thread_pool_) {
    process_chunks(0, 0, num_chunks);
    return;
  }

  // Distribute chunks across pool threads
  std::size_t chunks_per_thread =
      (num_chunks + effective_threads - 1) / effective_threads;

  thread_pool_->run_batch(
      [&](std::size_t t) {
        std::size_t start_chunk = t * chunks_per_thread;
        std::size_t end_chunk =
            std::min(start_chunk + chunks_per_thread, num_chunks);
        if (start_chunk < end_chunk) {
          process_chunks(t, start_chunk, end_chunk);
        }
      },
      effective_threads);
}

void BatchExecutionEngine::evaluate_map(const int *evidence_data,
                                        std::size_t batch_size,
                                        std::size_t num_vars,
                                        int *output_states) {
  validate_commercial_license();
  if (!output_states) {
    throw std::invalid_argument(
        "BatchExecutionEngine::evaluate_map received null output buffer.");
  }
  if (batch_size == 0) {
    return;
  }

  const std::size_t model_vars = jt_.graph()->num_variables();
  if (evidence_data && num_vars < model_vars) {
    throw std::invalid_argument(
        "BatchExecutionEngine::evaluate_map evidence width is smaller than "
        "the model variable count.");
  }

  const std::size_t num_chunks = (batch_size + chunk_size_ - 1) / chunk_size_;

  // Fast path for repeated scalar MAP queries (B=1).
  if (num_chunks == 1 && batch_size == 1) {
    if (!single_workspace_cache_ || single_workspace_batch_size_ != 1) {
      single_workspace_cache_ = std::make_unique<BatchWorkspace>(jt_, 1);
      single_workspace_batch_size_ = 1;
    } else {
      single_workspace_cache_->reset(1, 0);
    }

    BatchWorkspace &workspace = *single_workspace_cache_;
    if (evidence_data) {
      workspace.set_evidence_matrix(evidence_data, num_vars);
    } else {
      workspace.clear_evidence();
    }

    apply_soft_evidence_to_workspace(workspace, 0, 1);
    workspace.max_calibrate();
    workspace.query_map(output_states, model_vars);
    return;
  }

  // Multi-row MAP with thread pool
  auto process_chunks = [&](std::size_t thread_idx,
                            std::size_t start_chunk,
                            std::size_t end_chunk) {
    BatchWorkspace &workspace = get_pooled_workspace(thread_idx);

    for (std::size_t c = start_chunk; c < end_chunk; ++c) {
      const std::size_t chunk_start = c * chunk_size_;
      const std::size_t chunk_end =
          std::min(chunk_start + chunk_size_, batch_size);

      for (std::size_t row = chunk_start; row < chunk_end; ++row) {
        workspace.reset(1, row);
        const int *row_evidence =
            evidence_data ? (evidence_data + row * num_vars) : nullptr;
        if (row_evidence) {
          workspace.set_evidence_matrix(row_evidence, num_vars);
        } else {
          workspace.clear_evidence();
        }

        apply_soft_evidence_to_workspace(workspace, row, 1);
        workspace.max_calibrate();
        workspace.query_map(output_states + row * model_vars, model_vars);
      }
    }
  };

  const std::size_t effective_threads =
      std::max<std::size_t>(1, std::min(num_threads_, num_chunks));

  if (effective_threads == 1 || !thread_pool_) {
    process_chunks(0, 0, num_chunks);
    return;
  }

  const std::size_t chunks_per_thread =
      (num_chunks + effective_threads - 1) / effective_threads;

  thread_pool_->run_batch(
      [&](std::size_t t) {
        const std::size_t start_chunk = t * chunks_per_thread;
        const std::size_t end_chunk =
            std::min(start_chunk + chunks_per_thread, num_chunks);
        if (start_chunk < end_chunk) {
          process_chunks(t, start_chunk, end_chunk);
        }
      },
      effective_threads);
}

void BatchExecutionEngine::invalidate_workspace_cache() {
  single_workspace_cache_.reset();
  single_workspace_batch_size_ = 0;
  workspace_pool_.clear();
}

void BatchExecutionEngine::set_soft_evidence(bncore::NodeId var, const double *likelihoods, std::size_t n_states) {
  soft_evidence_scalar_[var].assign(likelihoods, likelihoods + n_states);
  soft_evidence_matrix_.erase(var);
}

void BatchExecutionEngine::set_soft_evidence_matrix(bncore::NodeId var, const double *likelihoods_matrix, std::size_t total_n_states) {
  soft_evidence_matrix_[var].assign(likelihoods_matrix, likelihoods_matrix + total_n_states);
  soft_evidence_scalar_.erase(var);
}

void BatchExecutionEngine::clear_soft_evidence() {
  soft_evidence_scalar_.clear();
  soft_evidence_matrix_.clear();
}

void BatchExecutionEngine::apply_soft_evidence_to_workspace(BatchWorkspace &workspace,
                                                            std::size_t current_batch_start,
                                                            std::size_t current_chunk_size) const {
  workspace.clear_soft_evidence();
  for (const auto &[nid, l_vec] : soft_evidence_scalar_) {
    workspace.set_soft_evidence(nid, l_vec.data(), l_vec.size());
  }
  for (const auto &[nid, l_mat] : soft_evidence_matrix_) {
    const std::size_t n_states_var = jt_.graph()->get_variable(nid).states.size();
    if (l_mat.size() < (current_batch_start + current_chunk_size) * n_states_var) {
      throw std::invalid_argument("Soft evidence matrix size is too small for the batch dimensions.");
    }
    workspace.set_soft_evidence_matrix(nid, l_mat.data() + current_batch_start * n_states_var, n_states_var);
  }
}

} // namespace bncore
