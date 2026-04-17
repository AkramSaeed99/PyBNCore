"""Validation tests for dynamic discretization (Neil-Tailor-Marquez MVP).

Each test compares the DD-computed posterior against a closed-form answer.
"""
import math

import numpy as np
import pytest
from pybncore._core import (
    DiscretizationManager,
    Graph,
    HybridEngine,
    HybridRunConfig,
)


def _normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def _lognormal_cdf(x, log_mu, log_sigma):
    if x <= 0.0:
        return 0.0
    return _normal_cdf(math.log(x), log_mu, log_sigma)


def _run_dd(graph, manager, queries, evidence=None, max_iters=8,
            eps_entropy=1e-4, eps_kl=1e-4):
    engine = HybridEngine(graph, manager, 1)
    cfg = HybridRunConfig()
    cfg.max_iters = max_iters
    cfg.eps_entropy = eps_entropy
    cfg.eps_kl = eps_kl
    ev = (np.asarray(evidence, dtype=np.int32)
          if evidence is not None
          else np.full(graph.num_variables(), -1, dtype=np.int32))
    qv = np.asarray(queries, dtype=np.int64)
    return engine.run(ev, qv, cfg)


# ---------------------------------------------------------------------------
# Test 1: Gaussian prior (single root node, no evidence, no parents).
# After convergence, the bin posterior should match integrated Normal mass.
# ---------------------------------------------------------------------------
def test_gaussian_prior_single_node():
    g = Graph()
    INIT_BINS = 8
    g.add_variable("X", [f"b{i}" for i in range(INIT_BINS)])

    dm = DiscretizationManager(max_bins_per_var=32)
    dm.register_normal(
        var_id=0, name="X", parents=[],
        mu_fn=lambda pb: 0.0,
        sigma_fn=lambda pb: 1.0,
        domain_lo=-4.0, domain_hi=4.0,
        initial_bins=INIT_BINS,
    )

    result = _run_dd(g, dm, queries=[0], max_iters=6)
    edges = np.asarray(result.edges[0])
    post = np.asarray(result.posteriors[0])

    # Posterior is properly normalised
    assert abs(post.sum() - 1.0) < 1e-6

    # Compare bin-by-bin against integrated closed-form Normal mass.
    # Mass per bin should match within coarse tolerance on clipped domain.
    expected = np.array(
        [_normal_cdf(edges[i + 1]) - _normal_cdf(edges[i])
         for i in range(len(edges) - 1)]
    )
    # Renormalise expected to the clipped domain (ignore mass outside [-4, 4]).
    expected /= expected.sum()
    # Max absolute diff on bin masses
    max_diff = np.max(np.abs(post - expected))
    assert max_diff < 0.05, f"max bin-mass diff = {max_diff:.4f}"


# ---------------------------------------------------------------------------
# Test 2: Gaussian chain with hard evidence on parent.
# X0 ~ N(0, 1), X1 = X0 + eps, eps ~ N(0, 0.1)  → X1|X0=x ~ N(x, 0.1)
# After evidence X0 is clamped to a bin whose midpoint is ~mu0, the
# posterior mean of X1 should be close to that midpoint.
# ---------------------------------------------------------------------------
def test_gaussian_chain_hard_evidence():
    g = Graph()
    INIT = 10
    g.add_variable("X0", [f"b{i}" for i in range(INIT)])
    g.add_variable("X1", [f"b{i}" for i in range(INIT)])
    g.add_edge("X0", "X1")

    dm = DiscretizationManager(max_bins_per_var=32)
    dm.register_normal(
        0, "X0", [],
        mu_fn=lambda pb: 0.0,
        sigma_fn=lambda pb: 1.0,
        domain_lo=-4.0, domain_hi=4.0,
        initial_bins=INIT,
    )
    # X1 | X0 ~ N(X0, 0.1)
    dm.register_normal(
        1, "X1", [0],
        mu_fn=lambda pb: pb.continuous_values[0],
        sigma_fn=lambda pb: 0.1,
        domain_lo=-4.0, domain_hi=4.0,
        initial_bins=INIT,
    )

    # Evidence: X0 fixed to bin containing ~0.0 (bin index varies by grid,
    # but at initial 10 bins over [-4, 4] each has width 0.8, bin 5 is [0, 0.8]
    # so its midpoint is 0.4).  Run w/o evidence on X0 first — confirm it
    # propagates.
    ev = np.full(2, -1, dtype=np.int32)
    ev[0] = 5  # observe X0 in bin [0, 0.8]; midpoint 0.4

    result = _run_dd(g, dm, queries=[0, 1], evidence=ev, max_iters=6)

    # X0's posterior should be a one-hot on bin 5 (or adjacent after refinement).
    post_x0 = np.asarray(result.posteriors[0])
    assert post_x0.max() > 0.99, f"X0 not clamped: {post_x0}"

    # X1's posterior mean should be close to the midpoint of X0's observed bin.
    edges_x1 = np.asarray(result.edges[1])
    post_x1 = np.asarray(result.posteriors[1])
    midpoints = 0.5 * (edges_x1[:-1] + edges_x1[1:])
    x1_mean = float(np.sum(post_x1 * midpoints))

    # Observed X0 bin midpoint
    edges_x0 = np.asarray(result.edges[0])
    obs_mid = 0.5 * (edges_x0[5] + edges_x0[6])

    # Low noise (0.1) ⇒ posterior mean should be very close to the parent midpoint
    assert abs(x1_mean - obs_mid) < 0.2, (
        f"X1 mean {x1_mean:.3f} vs observed parent midpoint {obs_mid:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 3: LogNormal reliability rare-event query.
# R ~ LogNormal(mu=-2, sigma=0.5), i.e. median ≈ 0.135.
# Query: P(R < 0.05).  Closed form = Phi((log(0.05) + 2) / 0.5).
# ---------------------------------------------------------------------------
def test_lognormal_reliability_tail():
    g = Graph()
    INIT = 12
    g.add_variable("R", [f"b{i}" for i in range(INIT)])

    dm = DiscretizationManager(max_bins_per_var=60)
    LOG_MU = -2.0
    LOG_SIGMA = 0.5
    dm.register_lognormal(
        0, "R", [],
        log_mu_fn=lambda pb: LOG_MU,
        log_sigma_fn=lambda pb: LOG_SIGMA,
        domain_lo=1e-4, domain_hi=5.0,
        initial_bins=INIT, log_spaced=True,
    )

    result = _run_dd(g, dm, queries=[0], max_iters=8,
                     eps_entropy=1e-5)

    edges = np.asarray(result.edges[0])
    post = np.asarray(result.posteriors[0])

    # Estimate P(R < 0.05) by summing bin masses fully below 0.05, plus a
    # proportional slice of the boundary bin.
    threshold = 0.05
    p_tail_dd = 0.0
    for j in range(len(post)):
        lo, hi = edges[j], edges[j + 1]
        if hi <= threshold:
            p_tail_dd += post[j]
        elif lo < threshold < hi:
            # Proportional fraction (linear within bin)
            p_tail_dd += post[j] * (threshold - lo) / (hi - lo)
            break

    p_tail_exact = _lognormal_cdf(threshold, LOG_MU, LOG_SIGMA)

    rel_err = abs(p_tail_dd - p_tail_exact) / max(p_tail_exact, 1e-12)
    # Rare-event tail estimation is fundamentally hard without
    # importance-sampling; 25% relative error at 60 bins is acceptable
    # for the MVP.  The order of magnitude must match exactly.
    assert rel_err < 0.25, (
        f"P(R<{threshold})  DD={p_tail_dd:.4e} "
        f"exact={p_tail_exact:.4e}  rel_err={rel_err:.2%}"
    )
    # Sanity: order of magnitude matches
    assert 0.5 * p_tail_exact < p_tail_dd < 2.0 * p_tail_exact


# ---------------------------------------------------------------------------
# Test 4: Convergence — running more iterations reduces the entropy error.
# Compare max-error after 1 iter vs 6 iters on the same network/seed.
# ---------------------------------------------------------------------------
def test_convergence_reduces_error():
    def run_with_iters(n):
        g = Graph()
        g.add_variable("X", [f"b{i}" for i in range(8)])
        dm = DiscretizationManager(max_bins_per_var=40)
        dm.register_normal(
            0, "X", [],
            mu_fn=lambda pb: 0.0,
            sigma_fn=lambda pb: 1.0,
            domain_lo=-4.0, domain_hi=4.0,
            initial_bins=8,
        )
        engine = HybridEngine(g, dm, 1)
        cfg = HybridRunConfig()
        cfg.max_iters = n
        cfg.eps_entropy = 1e-10  # force full iteration budget
        cfg.eps_kl = 1e-10
        ev = np.full(1, -1, dtype=np.int32)
        qv = np.asarray([0], dtype=np.int64)
        return engine.run(ev, qv, cfg).final_max_error

    err_1 = run_with_iters(1)
    err_6 = run_with_iters(6)
    # 6-iteration error must be strictly lower than 1-iteration error
    assert err_6 < err_1, (
        f"No improvement: err_1={err_1:.4e}, err_6={err_6:.4e}"
    )
    # And by a meaningful margin (at least 30% reduction)
    assert err_6 < err_1 * 0.7, (
        f"Convergence too slow: err_1={err_1:.4e}, err_6={err_6:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 5: Non-degenerate posterior after refinement (regression check).
# Ensure bins increase, edges get inserted at high-error regions.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test 6: Rare-event mode with threshold seeding sharply reduces tail error.
# Compare default LogNormal DD vs rare_event_mode=True + add_threshold(0.05).
# ---------------------------------------------------------------------------
def test_rare_event_mode_lowers_tail_error():
    LOG_MU, LOG_SIGMA = -2.0, 0.5
    THRESHOLD = 0.05
    p_exact = _lognormal_cdf(THRESHOLD, LOG_MU, LOG_SIGMA)

    def estimate_tail(rare_event_mode):
        g = Graph()
        g.add_variable("R", [f"b{i}" for i in range(12)])
        dm = DiscretizationManager(max_bins_per_var=60)
        dm.register_lognormal(
            0, "R", [],
            log_mu_fn=lambda pb: LOG_MU,
            log_sigma_fn=lambda pb: LOG_SIGMA,
            domain_lo=1e-4, domain_hi=5.0,
            initial_bins=12, log_spaced=True,
            rare_event_mode=rare_event_mode,
        )
        if rare_event_mode:
            dm.add_threshold(0, THRESHOLD)

        result = _run_dd(g, dm, queries=[0], max_iters=8, eps_entropy=1e-12)
        edges = np.asarray(result.edges[0])
        post = np.asarray(result.posteriors[0])

        p_tail = 0.0
        for j in range(len(post)):
            lo, hi = edges[j], edges[j + 1]
            if hi <= THRESHOLD:
                p_tail += post[j]
            elif lo < THRESHOLD < hi:
                p_tail += post[j] * (THRESHOLD - lo) / (hi - lo)
                break
        return p_tail

    p_default = estimate_tail(rare_event_mode=False)
    p_rare = estimate_tail(rare_event_mode=True)

    err_default = abs(p_default - p_exact) / p_exact
    err_rare = abs(p_rare - p_exact) / p_exact

    # Rare-event mode must be strictly more accurate
    assert err_rare < err_default, (
        f"rare_event_mode did not improve: default={err_default:.2%}, "
        f"rare={err_rare:.2%}"
    )
    # And below 5% absolute
    assert err_rare < 0.05, (
        f"rare-event tail accuracy still poor: {err_rare:.2%}"
    )


# ---------------------------------------------------------------------------
# Test 7: Continuous evidence by value works and updates across iterations.
# ---------------------------------------------------------------------------
def test_continuous_evidence_by_value():
    g = Graph()
    g.add_variable("X0", [f"b{i}" for i in range(10)])
    g.add_variable("X1", [f"b{i}" for i in range(10)])
    g.add_edge("X0", "X1")

    dm = DiscretizationManager(max_bins_per_var=30)
    dm.register_normal(
        0, "X0", [],
        mu_fn=lambda pb: 0.0,
        sigma_fn=lambda pb: 1.0,
        domain_lo=-4.0, domain_hi=4.0,
        initial_bins=10,
    )
    dm.register_normal(
        1, "X1", [0],
        mu_fn=lambda pb: pb.continuous_values[0],
        sigma_fn=lambda pb: 0.1,
        domain_lo=-4.0, domain_hi=4.0,
        initial_bins=10,
    )

    engine = HybridEngine(g, dm, 1)
    engine.set_evidence_continuous(0, 0.42)  # observe X0 = 0.42

    cfg = HybridRunConfig()
    cfg.max_iters = 6
    ev = np.full(2, -1, dtype=np.int32)
    qv = np.asarray([0, 1], dtype=np.int64)
    result = engine.run(ev, qv, cfg)

    # X0 should be pinned (one bin probability ≈ 1.0)
    post_x0 = np.asarray(result.posteriors[0])
    assert post_x0.max() > 0.99, f"X0 not pinned: {post_x0}"

    # X1 mean should be close to 0.42 (low noise)
    edges_x1 = np.asarray(result.edges[1])
    post_x1 = np.asarray(result.posteriors[1])
    midpoints = 0.5 * (edges_x1[:-1] + edges_x1[1:])
    x1_mean = float(np.sum(post_x1 * midpoints))
    assert abs(x1_mean - 0.42) < 0.15, (
        f"X1 mean {x1_mean:.3f} not close to evidence value 0.42"
    )


# ---------------------------------------------------------------------------
# Test 8: Deterministic SUM node — Z = X + Y.
# X, Y ~ N(0, 1) independent ⇒ Z ~ N(0, sqrt(2))
# ---------------------------------------------------------------------------
def test_deterministic_sum_node():
    g = Graph()
    INIT = 12
    g.add_variable("X", [f"b{i}" for i in range(INIT)])
    g.add_variable("Y", [f"b{i}" for i in range(INIT)])
    g.add_variable("Z", [f"b{i}" for i in range(INIT)])
    g.add_edge("X", "Z")
    g.add_edge("Y", "Z")

    dm = DiscretizationManager(max_bins_per_var=30)
    dm.register_normal(0, "X", [], lambda pb: 0.0, lambda pb: 1.0,
                        -4.0, 4.0, INIT)
    dm.register_normal(1, "Y", [], lambda pb: 0.0, lambda pb: 1.0,
                        -4.0, 4.0, INIT)
    dm.register_deterministic(
        2, "Z", [0, 1],
        fn=lambda pb: pb.continuous_values[0] + pb.continuous_values[1],
        domain_lo=-8.0, domain_hi=8.0, initial_bins=INIT,
        monotone=False, n_samples=64,
    )

    result = _run_dd(g, dm, queries=[2], max_iters=6)
    edges = np.asarray(result.edges[2])
    post = np.asarray(result.posteriors[2])
    midpoints = 0.5 * (edges[:-1] + edges[1:])

    z_mean = float(np.sum(post * midpoints))
    z_var = float(np.sum(post * (midpoints - z_mean) ** 2))

    assert abs(z_mean) < 0.20, f"Z mean = {z_mean:.3f}, expected ≈ 0"
    # Variance should be ≈ 2.0 (within 30% — discretization is coarse)
    assert 1.4 < z_var < 2.6, f"Z variance = {z_var:.3f}, expected ≈ 2.0"


# ---------------------------------------------------------------------------
# Test 9: Deterministic monotone function — Y = log(X).
# X ~ Uniform(1, 5), Y = log(X) ⇒ P(Y < 1) = P(X < e) = (e-1)/4 ≈ 0.4296
# ---------------------------------------------------------------------------
def test_deterministic_monotone_log():
    g = Graph()
    g.add_variable("X", [f"b{i}" for i in range(15)])
    g.add_variable("Y", [f"b{i}" for i in range(15)])
    g.add_edge("X", "Y")

    dm = DiscretizationManager(max_bins_per_var=40)
    dm.register_uniform(0, "X", [], lambda pb: 1.0, lambda pb: 5.0,
                         1.0, 5.0, 15)
    dm.register_deterministic(
        1, "Y", [0],
        fn=lambda pb: math.log(pb.continuous_values[0]),
        domain_lo=0.0, domain_hi=math.log(5.0) + 0.01,
        initial_bins=15, monotone=True,
    )

    result = _run_dd(g, dm, queries=[1], max_iters=6)
    edges_y = np.asarray(result.edges[1])
    post_y = np.asarray(result.posteriors[1])

    # P(Y < 1) by integrating posterior up to y=1
    threshold = 1.0
    p_dd = 0.0
    for j in range(len(post_y)):
        lo, hi = edges_y[j], edges_y[j + 1]
        if hi <= threshold:
            p_dd += post_y[j]
        elif lo < threshold < hi:
            p_dd += post_y[j] * (threshold - lo) / (hi - lo)
            break
    p_exact = (math.e - 1.0) / 4.0  # P(X < e) for X ~ U(1, 5)
    rel_err = abs(p_dd - p_exact) / p_exact
    assert rel_err < 0.05, (
        f"P(Y<1) DD={p_dd:.4f}, exact={p_exact:.4f}, rel_err={rel_err:.2%}"
    )


# ---------------------------------------------------------------------------
# Test 10: Threshold seeding inserts an exact edge at the threshold.
# ---------------------------------------------------------------------------
def test_threshold_seeding_exact_edge():
    g = Graph()
    g.add_variable("R", [f"b{i}" for i in range(8)])
    dm = DiscretizationManager(max_bins_per_var=20)
    dm.register_lognormal(
        0, "R", [], lambda pb: -2.0, lambda pb: 0.5,
        1e-4, 5.0, 8, log_spaced=True, rare_event_mode=True,
    )
    dm.add_threshold(0, 0.05)

    # The edge at 0.05 should already be present (no inference needed yet).
    edges = list(dm.variable_edges(0))
    assert any(abs(e - 0.05) < 1e-12 for e in edges), (
        f"threshold edge missing from initial grid: {edges}"
    )


def test_refinement_increases_bin_count():
    g = Graph()
    g.add_variable("X", [f"b{i}" for i in range(8)])
    dm = DiscretizationManager(max_bins_per_var=30)
    dm.register_normal(
        0, "X", [],
        mu_fn=lambda pb: 0.0,
        sigma_fn=lambda pb: 1.0,
        domain_lo=-4.0, domain_hi=4.0,
        initial_bins=8,
    )

    result = _run_dd(g, dm, queries=[0], max_iters=5, eps_entropy=1e-8)
    assert len(result.edges[0]) > 9, (
        f"Expected edges to grow past 9; got {len(result.edges[0])}"
    )
    post = np.asarray(result.posteriors[0])
    assert np.all(np.isfinite(post)), "Non-finite posterior"
    assert np.all(post >= 0.0), "Negative posterior"
    assert abs(post.sum() - 1.0) < 1e-6, "Posterior not normalised"
