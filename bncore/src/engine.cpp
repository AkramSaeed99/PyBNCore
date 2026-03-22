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
    : jt_(jt), num_threads_(num_threads), chunk_size_(chunk_size) {
  if (num_threads_ == 0)
    num_threads_ = std::thread::hardware_concurrency();
  if (num_threads_ == 0)
    num_threads_ = 1;
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
  // P2: Use query_marginals_multi — single clique scan for all queries,
  // with pre-baked state_of_dim maps (no Factor::marginalize allocation).
  // Convert size_t query_vars to NodeId (uint32_t) once here.
  std::vector<bncore::NodeId> qvars_nodeid(num_queries);
  for (std::size_t i = 0; i < num_queries; ++i)
    qvars_nodeid[i] = static_cast<bncore::NodeId>(query_vars[i]);

  auto write_query_outputs = [&](BatchWorkspace &workspace,
                                 std::size_t current_batch_start,
                                 std::size_t current_chunk_size) {
    double *out_row = output_data + current_batch_start * total_states;
    workspace.query_marginals_multi(qvars_nodeid.data(), num_queries,
                                    output_offsets, out_row);
  };


  // Fast path for the most common Python/HCL usage pattern:
  // single context query repeatedly from the same engine instance.
  if (num_chunks == 1) {
    const std::size_t current_chunk_size = batch_size;
    if (!single_workspace_cache_ ||
        single_workspace_batch_size_ != current_chunk_size) {
      single_workspace_cache_ =
          std::make_unique<BatchWorkspace>(jt_, current_chunk_size);
      single_workspace_batch_size_ = current_chunk_size;
    } else {
      single_workspace_cache_->reset(current_chunk_size);
    }

    BatchWorkspace &workspace = *single_workspace_cache_;
    const int *chunk_evidence = evidence_data;
    if (chunk_evidence) {
      workspace.set_evidence_matrix(chunk_evidence, num_vars);
    } else {
      workspace.clear_evidence();
    }
    workspace.calibrate();
    write_query_outputs(workspace, /*current_batch_start=*/0, current_chunk_size);
    return;
  }

  auto worker = [&](std::size_t start_chunk, std::size_t end_chunk) {
    BatchWorkspace workspace(jt_, chunk_size_);

    for (std::size_t c = start_chunk; c < end_chunk; ++c) {
      std::size_t current_batch_start = c * chunk_size_;
      std::size_t current_chunk_size =
          std::min(chunk_size_, batch_size - current_batch_start);

      workspace.reset(current_chunk_size);

      const int *chunk_evidence =
          evidence_data ? (evidence_data + current_batch_start * num_vars)
                        : nullptr;
      if (chunk_evidence)
        workspace.set_evidence_matrix(chunk_evidence, num_vars);

      workspace.calibrate();
      write_query_outputs(workspace, current_batch_start, current_chunk_size);
    }
  };

  const std::size_t effective_threads =
      std::max<std::size_t>(1, std::min(num_threads_, num_chunks));
  if (effective_threads == 1 || num_chunks <= 1) {
    worker(0, num_chunks);
    return;
  }

  std::vector<std::future<void>> futures;
  std::size_t chunks_per_thread =
      (num_chunks + effective_threads - 1) / effective_threads;

  for (std::size_t t = 0; t < effective_threads; ++t) {
    std::size_t start_chunk = t * chunks_per_thread;
    std::size_t end_chunk =
        std::min(start_chunk + chunks_per_thread, num_chunks);
    if (start_chunk < end_chunk) {
      futures.push_back(
          std::async(std::launch::async, worker, start_chunk, end_chunk));
    }
  }

  for (auto &f : futures) {
    f.get();
  }
}

void BatchExecutionEngine::invalidate_workspace_cache() {
  single_workspace_cache_.reset();
  single_workspace_batch_size_ = 0;
}

} // namespace bncore
