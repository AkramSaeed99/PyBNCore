#pragma once
#include "bncore/factors/factor.hpp"
#include "bncore/inference/junction_tree.hpp"
#include <unordered_map>
#include <vector>

namespace bncore {

// ---------------------------------------------------------------------------
// Pre-baked collect/distribute op for a single tree edge (child u → parent p).
// All index maps are computed once at construction; calibrate() does only
// table reads with zero heap allocation in the hot path.
// ---------------------------------------------------------------------------
struct CollectOp {
  std::size_t product_states;
  std::vector<uint32_t> pot_map;             // [prod_idx] → clique_pot flat idx
  std::vector<std::vector<uint32_t>> child_maps; // [child_i][prod_idx] → msg buf idx
  std::vector<std::size_t> child_cliques;    // parallel to child_maps
  std::vector<uint32_t> marg_map;            // [prod_idx] → up_msg buf idx
};

struct DistributeOp {
  std::size_t product_states;
  std::vector<uint32_t> pot_map;             // [prod_idx] → parent cliq pot idx
  bool has_parent_down = false;
  std::vector<uint32_t> down_map;            // [prod_idx] → down_msg[p] idx
  std::vector<std::vector<uint32_t>> sibling_maps;
  std::vector<std::size_t> sibling_cliques;
  std::vector<uint32_t> marg_map;            // [prod_idx] → down_msg[u] idx
  std::size_t parent_clique = 0;
};

struct AssembleOp {
  std::size_t product_states;
  bool has_down = false;
  std::vector<uint32_t> down_map;
  std::vector<std::vector<uint32_t>> child_maps;
  std::vector<std::size_t> child_cliques;
};

// ---------------------------------------------------------------------------
// Pre-baked per-variable marginalisation map for fast query extraction.
// query_state_map[node_dim][entry_in_clique] = state index of that variable.
// This replaces Factor::marginalize() with a direct accumulation loop.
// ---------------------------------------------------------------------------
struct CliqueMargInfo {
  // For each variable position in the clique scope:
  //   state_of_dim[d][flat_idx] = coord of scope[d] at flat index flat_idx
  // Length: [scope_size][clique_states]
  // Stored as uint8_t since state counts for binary/ternary vars are tiny.
  std::vector<std::vector<uint8_t>> state_of_dim;
  std::size_t n_states_total = 0;  // clique_states_[ci]
};

class BatchWorkspace {
public:
  explicit BatchWorkspace(const JunctionTree &jt, std::size_t batch_size = 1);

  void reset(std::size_t new_batch_size);

  void set_evidence_matrix(const int *evidence_matrix, std::size_t num_vars);
  void clear_evidence();

  void calibrate();

  // Single-variable marginal (kept for backward compat with engine fallback)
  DenseTensor query_marginal(NodeId node) const;

  // Multi-variable marginal — one scan of each relevant clique.
  // out is flat: [batch_size × total_states], where states are packed per
  // query variable as specified by offsets[0..num_queries].
  void query_marginals_multi(const NodeId *query_vars, std::size_t num_queries,
                             const std::size_t *offsets,
                             double *out) const;

private:
  void rebuild_clique_potentials(std::size_t target_batch_size);
  void snapshot_base_potentials();
  void build_node_to_clique_map();
  void build_message_schedule();

  const JunctionTree &jt_;
  std::size_t batch_size_;

  std::vector<Factor> clique_potentials_;
  std::vector<Factor> base_clique_potentials_;
  const int *evidence_matrix_ = nullptr;
  std::size_t evidence_num_vars_ = 0;

  std::size_t root_clique_;
  std::vector<std::size_t> collect_order_;
  std::vector<std::size_t> distribute_order_;
  std::vector<std::size_t> parent_of_;
  std::vector<std::vector<std::size_t>> children_of_;

  // Pre-baked zero-allocation message schedule
  std::vector<std::vector<double>> up_msg_buf_;
  std::vector<std::vector<double>> down_msg_buf_;
  std::vector<std::size_t> up_msg_size_;
  std::vector<std::size_t> down_msg_size_;
  std::vector<std::size_t> sepset_size_;
  std::vector<std::size_t> clique_states_;

  std::vector<CollectOp>    collect_ops_;
  std::vector<DistributeOp> distribute_ops_;
  std::vector<AssembleOp>   assemble_ops_;

  // P1: Persistent scratch buffer — sized to max product_states at construction,
  // reused every calibrate() with zero heap allocation in the hot path.
  std::vector<double> scratch_buf_;
  std::size_t max_product_states_ = 0;

  // Calibrated potentials
  std::vector<std::vector<double>> cal_pot_buf_;
  std::vector<std::size_t> cal_pot_size_;

  // Base (no-evidence) caches for lazy propagation
  std::vector<std::vector<double>> base_up_buf_;
  std::vector<std::vector<double>> base_down_buf_;
  std::vector<std::vector<double>> base_cal_buf_;
  bool is_first_calibration_ = true;

  // P2: Pre-baked per-clique per-variable marginalisation maps
  std::vector<CliqueMargInfo> clique_marg_info_;

  // P3: Per-clique evidence scratch buffers (replaces fixed-size BumpAllocator)
  // ev_scratch_[ci] is lazily sized to clique_states_[ci] on first evidence use.
  std::vector<std::vector<double>> ev_scratch_;
  // Tracks which clique entries are owned by ev_scratch_ (not base potentials)
  std::vector<bool> ev_scratch_active_;

  // O(1) lookups
  std::unordered_map<NodeId, std::size_t> node_to_clique_;
  std::unordered_map<NodeId, std::vector<std::size_t>> node_in_cliques_;
};

} // namespace bncore
