#pragma once
#include "bncore/factors/factor.hpp"
#include "bncore/inference/junction_tree.hpp"
#include "bncore/util/bump_allocator.hpp"
#include <unordered_map>
#include <vector>

namespace bncore {

class BatchWorkspace {
public:
  explicit BatchWorkspace(const JunctionTree &jt, std::size_t batch_size = 1);

  void reset(std::size_t new_batch_size);

  void set_evidence_matrix(const int *evidence_matrix, std::size_t num_vars);
  void clear_evidence();

  void calibrate();

  DenseTensor query_marginal(NodeId node) const;

private:
  void rebuild_clique_potentials(std::size_t target_batch_size);
  void snapshot_base_potentials();

  const JunctionTree &jt_;
  std::size_t batch_size_;
  BumpAllocator allocator_;

  std::vector<Factor> clique_potentials_;
  std::vector<Factor> base_clique_potentials_;
  const int *evidence_matrix_ = nullptr;
  std::size_t evidence_num_vars_ = 0;

  std::size_t root_clique_;
  std::vector<std::size_t> collect_order_;
  std::vector<std::size_t> distribute_order_;
  std::vector<std::size_t> parent_of_;
  std::vector<std::vector<std::size_t>> children_of_;

  std::vector<Factor> up_messages_;
  std::vector<Factor> down_messages_;
  std::vector<Factor> calibrated_potentials_;
};

} // namespace bncore
