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

  void reset(std::size_t new_batch_size, std::size_t cpt_batch_offset = 0);

  // ---------------------------------------------------------------------------
  // Hard evidence (existing API — unchanged semantics)
  // ---------------------------------------------------------------------------
  void set_evidence_matrix(const int *evidence_matrix, std::size_t num_vars);
  void clear_evidence();

  // ---------------------------------------------------------------------------
  // Soft / virtual evidence
  //
  // set_soft_evidence: supply a likelihood vector lambda[0..n_states-1] for a
  // single variable.  lambda[s] = P(measurement | X=s).  Hard evidence state s0
  // is a special case: lambda[s0]=1, all others 0.
  //
  // n_states must match the variable's cardinality; throws otherwise.
  // Likelihoods need not sum to one (improper likelihoods are valid).
  // Likelihoods must be non-negative; throws on negative entries.
  //
  // set_soft_evidence_matrix: batched variant — likelihoods[b * n_states + s]
  // is the likelihood of state s for batch row b.  Shape: [B × n_states].
  // ---------------------------------------------------------------------------
  void set_soft_evidence(NodeId var, const double *likelihoods,
                         std::size_t n_states);
  void set_soft_evidence_matrix(NodeId var, const double *likelihoods_matrix,
                                 std::size_t n_states);
  void clear_soft_evidence();

  // ---------------------------------------------------------------------------
  // Sum-product calibration (standard marginal inference)
  // ---------------------------------------------------------------------------
  void calibrate();

  // ---------------------------------------------------------------------------
  // Max-product calibration (MAP / MPE inference)
  //
  // After max_calibrate(), call query_map() to decode the joint MAP assignment.
  // ---------------------------------------------------------------------------
  void max_calibrate();

  // Writes the most probable state index for each variable into out_states
  // (length n_vars = JT graph variable count).  Variables not in the JT are
  // written as -1.  Array must be pre-allocated by the caller.
  void query_map(int *out_states, std::size_t n_vars) const;

  // ---------------------------------------------------------------------------
  // Query extraction (sum-product results)
  // ---------------------------------------------------------------------------

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
  void resize_runtime_buffers(std::size_t target_batch_size);
  void snapshot_base_potentials();
  void build_node_to_clique_map();
  void build_message_schedule();

  // Internal helpers
  // apply_soft_evidence_to_clique: applies stored soft/hard likelihood vectors
  // to a single clique potential.  Used by both calibrate() paths.
  // Returns true if any likelihood was applied to this clique.
  bool apply_soft_evidence_to_clique(std::size_t ci);

  const JunctionTree &jt_;
  std::size_t batch_size_;
  std::size_t cpt_batch_offset_ = 0;

  std::vector<Factor> clique_potentials_;
  std::vector<Factor> base_clique_potentials_;
  std::size_t base_batch_size_ = 0;
  std::size_t base_batch_offset_ = 0;
  bool base_depends_on_offset_ = false;
  const int *evidence_matrix_ = nullptr;
  std::size_t evidence_num_vars_ = 0;

  // Soft evidence storage:
  //   soft_evidence_scalar_[var] = likelihood vector for B=1 or shared across B
  //   soft_evidence_matrix_[var] = per-row likelihood matrix [B * n_states]
  //                                (populated by set_soft_evidence_matrix)
  std::unordered_map<NodeId, std::vector<double>> soft_evidence_scalar_;
  std::unordered_map<NodeId, std::vector<double>> soft_evidence_matrix_;

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
  // then resized as max_product_states * batch_size for runtime chunk geometry.
  // Reused every calibrate() with zero heap allocation in the hot path.
  std::vector<double> scratch_buf_;
  std::size_t max_product_states_ = 0;

  // Phase-A: workspace-owned uint8_t scratch flags — pre-sized at construction,
  // cleared via memset at top of calibrate() with ZERO heap allocation.
  // Using uint8_t instead of vector<bool>: byte-addressable, not bit-packed.
  std::vector<uint8_t> ev_clique_;    // did evidence touch this clique?
  std::vector<uint8_t> up_changed_;   // did up_msg change in collect?
  std::vector<uint8_t> down_changed_; // did down_msg change in distribute?
  std::vector<uint8_t> cal_changed_;  // does assembled cal_pot need update?

  // Calibrated potentials (sum-product)
  std::vector<std::vector<double>> cal_pot_buf_;
  std::vector<std::size_t> cal_pot_size_;

  // Max-product message buffers and traceback (MAP inference)
  // Layout mirrors up_msg_buf_: [sepset_size * B]
  std::vector<std::vector<double>> map_up_msg_buf_;
  // map_traceback_[clique][product_state] = which child-combined input index
  // achieved the max; used by query_map() to decode argmax state per variable.
  std::vector<std::vector<uint32_t>> map_traceback_;
  // Calibrated max-product beliefs per clique (assembled after max_calibrate)
  std::vector<std::vector<double>> map_cal_pot_buf_;

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

  // Query-time scratch for safe fallback when clique count exceeds stack guard.
  mutable std::vector<uint8_t> query_visited_scratch_;
};

} // namespace bncore
