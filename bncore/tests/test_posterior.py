"""Unit tests for ContinuousPosterior.  No graph / engine required."""
import numpy as np
import pytest

from pybncore.posterior import ContinuousPosterior


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------
def test_construction_renormalises_masses():
    p = ContinuousPosterior("X", edges=[0, 1, 2], bin_masses=[2.0, 2.0])
    assert np.isclose(p.bin_masses.sum(), 1.0)


def test_construction_rejects_non_monotonic_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        ContinuousPosterior("X", edges=[0, 2, 1], bin_masses=[0.5, 0.5])


def test_construction_rejects_negative_mass():
    with pytest.raises(ValueError, match="non-negative"):
        ContinuousPosterior("X", edges=[0, 1, 2], bin_masses=[1.1, -0.1])


def test_construction_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="bin_masses length"):
        ContinuousPosterior("X", edges=[0, 1, 2], bin_masses=[0.5, 0.3, 0.2])


# ---------------------------------------------------------------------------
# Summary statistics — uniform case (closed-form)
# ---------------------------------------------------------------------------
def test_mean_variance_uniform():
    # Uniform on [0, 10] with 10 equal bins: mean = 5, var = 100/12 ≈ 8.33
    edges = np.linspace(0, 10, 11)
    masses = np.full(10, 0.1)
    p = ContinuousPosterior("X", edges, masses)
    assert np.isclose(p.mean(), 5.0, atol=1e-10)
    # Using midpoints (not true var): E[mid^2] - E[mid]^2
    mids = 0.5 * (edges[:-1] + edges[1:])
    expected_var = float(np.mean(mids ** 2) - np.mean(mids) ** 2)
    assert np.isclose(p.variance(), expected_var, atol=1e-10)


def test_mode_bin_argmax():
    p = ContinuousPosterior("X", edges=[0, 1, 2, 3, 4],
                             bin_masses=[0.1, 0.5, 0.3, 0.1])
    assert p.mode_bin() == 1


# ---------------------------------------------------------------------------
# Probability queries
# ---------------------------------------------------------------------------
def test_cdf_clamps_outside_support():
    p = ContinuousPosterior("X", edges=[0, 1, 2], bin_masses=[0.5, 0.5])
    assert p.cdf(-10.0) == 0.0
    assert p.cdf(99.0) == 1.0


def test_cdf_linear_within_bin():
    # Uniform on [0, 2]: CDF should be linear
    p = ContinuousPosterior("X", edges=[0, 2], bin_masses=[1.0])
    assert np.isclose(p.cdf(0.0), 0.0)
    assert np.isclose(p.cdf(0.5), 0.25)
    assert np.isclose(p.cdf(1.0), 0.5)
    assert np.isclose(p.cdf(2.0), 1.0)


def test_prob_between():
    p = ContinuousPosterior("X", edges=[0, 1, 2, 3], bin_masses=[0.1, 0.6, 0.3])
    # P(0.5 <= X < 2.5) = half of bin 0 + all of bin 1 + half of bin 2
    expected = 0.5 * 0.1 + 0.6 + 0.5 * 0.3
    assert np.isclose(p.prob_between(0.5, 2.5), expected)


def test_prob_between_rejects_reversed_bounds():
    p = ContinuousPosterior("X", edges=[0, 1], bin_masses=[1.0])
    with pytest.raises(ValueError, match="a <= b"):
        p.prob_between(0.8, 0.2)


def test_prob_greater_than_is_complement_of_cdf():
    p = ContinuousPosterior("X", edges=[0, 1, 2, 3],
                             bin_masses=[0.2, 0.5, 0.3])
    for x in [-1.0, 0.5, 1.0, 2.0, 4.0]:
        assert np.isclose(p.cdf(x) + p.prob_greater_than(x), 1.0)


# ---------------------------------------------------------------------------
# Quantile (inverse CDF)
# ---------------------------------------------------------------------------
def test_quantile_clamps_at_boundaries():
    p = ContinuousPosterior("X", edges=[0, 10], bin_masses=[1.0])
    assert p.quantile(-0.5) == 0.0
    assert p.quantile(1.5) == 10.0


def test_quantile_uniform_linear():
    p = ContinuousPosterior("X", edges=[0, 10], bin_masses=[1.0])
    assert np.isclose(p.quantile(0.5), 5.0)
    assert np.isclose(p.quantile(0.1), 1.0)
    assert np.isclose(p.quantile(0.9), 9.0)


def test_quantile_inverse_of_cdf():
    """quantile(cdf(x)) ≈ x for x in the support."""
    edges = np.array([0, 1, 3, 5, 10], dtype=float)
    masses = np.array([0.1, 0.3, 0.4, 0.2])
    p = ContinuousPosterior("X", edges, masses)
    for x in [0.5, 1.5, 3.5, 6.0, 8.0]:
        assert np.isclose(p.quantile(p.cdf(x)), x, atol=1e-10), x


def test_median_equals_quantile_half():
    p = ContinuousPosterior("X", edges=[0, 1, 2, 3], bin_masses=[0.2, 0.5, 0.3])
    assert np.isclose(p.median(), p.quantile(0.5))


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def test_pdf_is_mass_over_width():
    p = ContinuousPosterior("X", edges=[0, 1, 3], bin_masses=[0.4, 0.6])
    assert np.isclose(p.pdf(0.5), 0.4 / 1.0)
    assert np.isclose(p.pdf(2.0), 0.6 / 2.0)


def test_pdf_zero_outside_support():
    p = ContinuousPosterior("X", edges=[0, 1], bin_masses=[1.0])
    assert p.pdf(-1.0) == 0.0
    assert p.pdf(2.0) == 0.0


def test_pdf_integrates_to_one():
    """∫ pdf(x) dx over support == 1.0 via midpoint rule on edges."""
    edges = np.array([0, 0.5, 2.0, 5.0, 5.1])
    masses = np.array([0.1, 0.3, 0.5, 0.1])
    p = ContinuousPosterior("X", edges, masses)
    # Analytic: sum of bin_mass is 1 by construction; pdf integral is the same.
    integral = sum(p.pdf(0.5 * (edges[j] + edges[j + 1])) *
                    (edges[j + 1] - edges[j]) for j in range(len(masses)))
    assert np.isclose(integral, 1.0)


# ---------------------------------------------------------------------------
# Repr and shape properties
# ---------------------------------------------------------------------------
def test_repr_contains_key_fields():
    p = ContinuousPosterior("Temp", edges=[0, 1, 2], bin_masses=[0.5, 0.5])
    r = repr(p)
    assert "Temp" in r
    assert "num_bins=2" in r
    assert "mean=" in r


def test_len_is_num_bins():
    p = ContinuousPosterior("X", edges=[0, 1, 2, 3], bin_masses=[0.1, 0.4, 0.5])
    assert len(p) == 3
    assert p.num_bins == 3


def test_support_matches_edges():
    p = ContinuousPosterior("X", edges=[-2, 0, 5], bin_masses=[0.3, 0.7])
    assert p.support == (-2.0, 5.0)


# ---------------------------------------------------------------------------
# Plotting graceful fallback
# ---------------------------------------------------------------------------
def test_plot_returns_none_if_matplotlib_missing(monkeypatch):
    import builtins
    orig_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("mocked absence")
        return orig_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    p = ContinuousPosterior("X", edges=[0, 1, 2], bin_masses=[0.5, 0.5])
    assert p.plot() is None
