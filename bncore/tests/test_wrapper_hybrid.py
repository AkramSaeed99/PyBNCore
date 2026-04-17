"""Tests for PyBNCoreWrapper's hybrid (continuous-variable) API.

These exercise the high-level wrapper surface — `add_normal`, `hybrid_query`,
etc.  Lower-level validation of the DD engine itself lives in
`tests/test_dynamic_discretization.py`.
"""
import math

import numpy as np
import pytest

from pybncore import ContinuousPosterior, PyBNCoreWrapper


def _normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


# ---------------------------------------------------------------------------
# Registration + basic single-node posterior
# ---------------------------------------------------------------------------
def test_add_normal_prior_single_node():
    w = PyBNCoreWrapper()
    w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=8)
    result = w.hybrid_query(["X"], max_iters=6)

    assert "X" in result
    post = result["X"]
    assert isinstance(post, ContinuousPosterior)
    # Normal(0,1): mean ≈ 0, std ≈ 1 within discretization error
    assert abs(post.mean()) < 0.05
    assert abs(post.std() - 1.0) < 0.1
    # Tail probability matches Phi
    assert abs(post.prob_less_than(-1.0) - _normal_cdf(-1.0)) < 0.01


def test_hybrid_result_exposes_convergence_diagnostics():
    def _fresh_run(max_iters):
        w = PyBNCoreWrapper()
        w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=8)
        return w.hybrid_query(
            ["X"], max_iters=max_iters,
            eps_entropy=1e-20, eps_kl=1e-20,
        )

    r1 = _fresh_run(1)
    assert r1.iterations_used == 1
    assert r1.final_max_error > 0
    assert r1.converged is False

    # On a fresh wrapper, more iterations strictly reduces the error
    r6 = _fresh_run(6)
    assert r6.final_max_error < r1.final_max_error


# ---------------------------------------------------------------------------
# Mixed discrete/continuous model — return types match declared kinds
# ---------------------------------------------------------------------------
def test_hybrid_query_returns_dict_for_discrete():
    # Discrete parent D with two states, continuous child C whose mean
    # depends on D.  Query both.
    w = PyBNCoreWrapper()
    # Add a discrete variable via the graph-level API
    from pybncore._core import Graph
    if w._graph is None:
        w._graph = Graph()
    w._graph.add_variable("D", ["zero", "one"])
    w._graph.set_cpt("D", np.array([0.4, 0.6]))
    w._cache_metadata()

    # Continuous child conditioned on the discrete parent
    w.add_normal(
        "C", parents=["D"],
        mu=lambda d: 0.0 if d == 0 else 5.0,
        sigma=1.0,
        domain=(-5.0, 10.0), initial_bins=10,
    )

    result = w.hybrid_query(["C", "D"], max_iters=4)
    assert isinstance(result["C"], ContinuousPosterior)
    # Discrete variables return a {state: prob} dict (same as batch_query_marginals)
    assert isinstance(result["D"], dict)
    assert set(result["D"].keys()) == {"zero", "one"}
    assert abs(sum(result["D"].values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Evidence by value — continuous
# ---------------------------------------------------------------------------
def test_set_continuous_evidence_shifts_posterior():
    w = PyBNCoreWrapper()
    w.add_normal("X0", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=10)
    w.add_normal(
        "X1", parents=["X0"],
        mu=lambda x0: x0, sigma=0.1,
        domain=(-4.0, 4.0), initial_bins=10,
    )

    w.set_continuous_evidence({"X0": 0.42})
    result = w.hybrid_query(["X0", "X1"], max_iters=5)

    # X0 is pinned — one bin has ~all the mass
    assert result["X0"].bin_masses.max() > 0.99
    # X1's posterior mean should be close to 0.42 (low noise)
    assert abs(result["X1"].mean() - 0.42) < 0.2


def test_clear_continuous_evidence_restores_prior():
    w = PyBNCoreWrapper()
    w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=8)
    w.set_continuous_evidence({"X": 2.5})
    res_with = w.hybrid_query(["X"], max_iters=3)
    assert res_with["X"].mean() > 2.0
    w.clear_continuous_evidence()
    res_cleared = w.hybrid_query(["X"], max_iters=3)
    assert abs(res_cleared["X"].mean()) < 0.2  # back to prior ≈ 0


# ---------------------------------------------------------------------------
# Deterministic functional node
# ---------------------------------------------------------------------------
def test_deterministic_sum_node():
    """X + Y with X, Y ~ N(0,1) independent → Z ~ N(0, sqrt(2))."""
    w = PyBNCoreWrapper()
    w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=12)
    w.add_normal("Y", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=12)
    w.add_deterministic(
        "Z", parents=["X", "Y"],
        fn=lambda x, y: x + y,
        domain=(-8.0, 8.0), initial_bins=12, n_samples=64,
    )
    result = w.hybrid_query(["Z"], max_iters=4)
    assert abs(result["Z"].mean()) < 0.2
    assert 1.4 < result["Z"].variance() < 2.6


# ---------------------------------------------------------------------------
# Rare-event mode + threshold seeding
# ---------------------------------------------------------------------------
def test_lognormal_rare_event_mode_sub_1pct_tail_error():
    def _lognormal_cdf(x, log_mu, log_sigma):
        if x <= 0:
            return 0.0
        return _normal_cdf(math.log(x), log_mu, log_sigma)

    w = PyBNCoreWrapper()
    LOG_MU, LOG_SIGMA, THRESHOLD = -2.0, 0.5, 0.07
    w.add_lognormal(
        "R", log_mu=LOG_MU, log_sigma=LOG_SIGMA,
        domain=(1e-4, 10.0), initial_bins=12,
        log_spaced=True, rare_event_mode=True,
    )
    w.add_threshold("R", THRESHOLD)

    result = w.hybrid_query(["R"], max_iters=6)
    p_tail = result["R"].prob_less_than(THRESHOLD)
    p_exact = _lognormal_cdf(THRESHOLD, LOG_MU, LOG_SIGMA)
    rel_err = abs(p_tail - p_exact) / p_exact
    assert rel_err < 0.01, f"rel_err = {rel_err:.2%}"


# ---------------------------------------------------------------------------
# Interop + error contracts
# ---------------------------------------------------------------------------
def test_batch_query_marginals_on_hybrid_raises_helpful_error():
    w = PyBNCoreWrapper()
    w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=8)
    with pytest.raises(RuntimeError, match="hybrid_query"):
        w.batch_query_marginals(["X"])


def test_add_threshold_rejects_discrete():
    w = PyBNCoreWrapper()
    w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=8)
    with pytest.raises(ValueError, match="unknown"):
        w.add_threshold("nope", 0.5)


def test_hybrid_query_without_continuous_vars_errors():
    w = PyBNCoreWrapper()
    # No continuous vars registered
    with pytest.raises(RuntimeError, match="requires at least one continuous"):
        w.hybrid_query(["X"])


def test_lognormal_domain_validation():
    w = PyBNCoreWrapper()
    with pytest.raises(ValueError, match="domain_lo > 0"):
        w.add_lognormal(
            "R", log_mu=0.0, log_sigma=1.0,
            domain=(0.0, 10.0),  # invalid: must be > 0
        )


def test_add_normal_initial_bins_validation():
    w = PyBNCoreWrapper()
    with pytest.raises(ValueError, match="initial_bins"):
        w.add_normal("X", mu=0.0, sigma=1.0, domain=(-1, 1), initial_bins=1)


def test_discrete_model_still_works_unchanged():
    """Regression: existing discrete API is not broken by the hybrid additions."""
    from pybncore._core import Graph

    w = PyBNCoreWrapper()
    g = Graph()
    g.add_variable("A", ["True", "False"])
    g.add_variable("B", ["High", "Low"])
    g.add_edge("A", "B")
    g.set_cpt("A", np.array([0.7, 0.3]))
    g.set_cpt("B", np.array([0.9, 0.1, 0.2, 0.8]))

    w._graph = g
    w._cache_metadata()

    marginals = w.batch_query_marginals(["A", "B"])
    assert np.isclose(marginals["A"]["True"], 0.7)
    assert np.isclose(
        marginals["B"]["High"], 0.7 * 0.9 + 0.3 * 0.2, atol=1e-10
    )
