#include "bncore/inference/engine.hpp"
#include <algorithm>

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
  std::size_t num_chunks = (batch_size + chunk_size_ - 1) / chunk_size_;

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

      DenseTensor marginal = workspace.query_marginal(query_var);
      std::size_t num_states = marginal.shape()[0];

      for (std::size_t b = 0; b < current_chunk_size; ++b) {
        double sum = 0.0;
        for (std::size_t s = 0; s < num_states; ++s) {
          sum += marginal.data()[s * current_chunk_size + b];
        }
        for (std::size_t s = 0; s < num_states; ++s) {
          output_data[(current_batch_start + b) * num_states + s] =
              (sum > 0.0) ? (marginal.data()[s * current_chunk_size + b] / sum)
                          : 0.0;
        }
      }
    }
  };

  std::vector<std::future<void>> futures;
  std::size_t chunks_per_thread =
      (num_chunks + num_threads_ - 1) / num_threads_;

  for (std::size_t t = 0; t < num_threads_; ++t) {
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

} // namespace bncore
