"""Posterior value classes for hybrid / continuous inference.

A :class:`ContinuousPosterior` represents the bin-based posterior
distribution of a continuous variable after dynamic discretization.
It exposes the common summary statistics (mean, variance, quantiles)
and probability queries (CDF, P(a < X < b)) that reliability / PRA
workflows need, so users don't have to reimplement them every time.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

__all__ = ["ContinuousPosterior"]


class ContinuousPosterior:
    """Bin-based posterior over a continuous variable.

    The variable's support is partitioned into ``m`` bins, with edges
    ``edges[0] < edges[1] < ... < edges[m]``.  ``bin_masses[j]`` is the
    posterior probability ``P(X in [edges[j], edges[j+1]))`` and the
    masses sum to 1 (within numerical tolerance).

    Bin densities are computed as ``bin_masses[j] / (edges[j+1] - edges[j])``
    and assumed piecewise-constant; summary statistics and quantiles use
    the bin midpoint as the representative value.

    Parameters
    ----------
    name : str
        Variable name (for debugging / plotting labels).
    edges : array-like, shape (m+1,)
        Bin boundaries, strictly increasing.
    bin_masses : array-like, shape (m,)
        Posterior probability per bin; must be non-negative.  The array is
        renormalised to sum to 1 if its sum is positive.

    Raises
    ------
    ValueError
        If ``edges`` is not strictly increasing, ``bin_masses`` has the
        wrong length, or contains negative values.

    Examples
    --------
    >>> import numpy as np
    >>> p = ContinuousPosterior("X", edges=[0, 1, 2, 3], bin_masses=[0.1, 0.6, 0.3])
    >>> round(p.mean(), 3)
    1.7
    >>> round(p.prob_between(0.5, 2.5), 3)
    0.8
    """

    __slots__ = ("name", "edges", "bin_masses")

    def __init__(self, name: str,
                 edges: "np.ndarray | list[float]",
                 bin_masses: "np.ndarray | list[float]") -> None:
        self.name = str(name)
        e = np.asarray(edges, dtype=np.float64)
        m = np.asarray(bin_masses, dtype=np.float64)
        if e.ndim != 1 or e.size < 2:
            raise ValueError("edges must be a 1D array with >= 2 entries")
        if m.ndim != 1:
            raise ValueError("bin_masses must be 1D")
        if m.size + 1 != e.size:
            raise ValueError(
                f"bin_masses length ({m.size}) must equal len(edges)-1 ({e.size - 1})"
            )
        if np.any(np.diff(e) <= 0):
            raise ValueError("edges must be strictly increasing")
        if np.any(m < 0):
            raise ValueError("bin_masses must be non-negative")
        total = float(m.sum())
        if total > 0:
            m = m / total
        self.edges = e
        self.bin_masses = m

    # ---------------------------------------------------------------- shape
    @property
    def num_bins(self) -> int:
        return int(self.bin_masses.size)

    @property
    def support(self) -> "tuple[float, float]":
        """Half-open support ``[edges[0], edges[-1])``."""
        return float(self.edges[0]), float(self.edges[-1])

    # --------------------------------------------------------- summary stats
    def _midpoints(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def mean(self) -> float:
        """Posterior mean (expectation using bin midpoints)."""
        return float(np.sum(self._midpoints() * self.bin_masses))

    def variance(self) -> float:
        """Posterior variance (biased; treats bins as point masses at midpoints)."""
        mp = self._midpoints()
        mu = float(np.sum(mp * self.bin_masses))
        return float(np.sum(self.bin_masses * (mp - mu) ** 2))

    def std(self) -> float:
        return float(np.sqrt(max(self.variance(), 0.0)))

    def mode_bin(self) -> int:
        """Return the index of the bin with the highest probability mass."""
        return int(np.argmax(self.bin_masses))

    # ------------------------------------------------------ probability queries
    def cdf(self, x: float) -> float:
        """Return ``P(X < x)`` via linear interpolation within the bin
        containing ``x``.  Values outside the support clamp to 0 or 1."""
        x = float(x)
        if x <= self.edges[0]:
            return 0.0
        if x >= self.edges[-1]:
            return 1.0
        # Find the bin containing x
        j = int(np.searchsorted(self.edges, x, side="right") - 1)
        j = max(0, min(j, self.num_bins - 1))
        # Mass below the containing bin
        below = float(np.sum(self.bin_masses[:j]))
        # Fractional mass inside the containing bin (linear within bin)
        lo, hi = self.edges[j], self.edges[j + 1]
        frac = (x - lo) / (hi - lo) if hi > lo else 0.0
        return below + float(self.bin_masses[j]) * frac

    def prob_less_than(self, x: float) -> float:
        """Alias for :meth:`cdf`."""
        return self.cdf(x)

    def prob_greater_than(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def prob_between(self, a: float, b: float) -> float:
        """Return ``P(a <= X < b)``.  Ordered: requires ``a <= b``."""
        if b < a:
            raise ValueError(f"prob_between requires a <= b (got a={a}, b={b})")
        return self.cdf(b) - self.cdf(a)

    def quantile(self, q: float) -> float:
        """Return ``x`` such that ``P(X < x) = q`` via linear interpolation.

        Clamps to the support boundaries for ``q <= 0`` or ``q >= 1``.
        """
        q = float(q)
        if q <= 0.0:
            return float(self.edges[0])
        if q >= 1.0:
            return float(self.edges[-1])
        # Cumulative mass at each edge
        cum = np.concatenate([[0.0], np.cumsum(self.bin_masses)])
        # Find the bin where the CDF first reaches or exceeds q
        j = int(np.searchsorted(cum, q, side="left") - 1)
        j = max(0, min(j, self.num_bins - 1))
        # Linear interp inside bin j: cum[j] + frac * bin_mass[j] = q
        mass = self.bin_masses[j]
        if mass <= 0.0:
            return float(self.edges[j])
        frac = (q - cum[j]) / mass
        frac = max(0.0, min(1.0, frac))
        return float(self.edges[j] + frac * (self.edges[j + 1] - self.edges[j]))

    def median(self) -> float:
        return self.quantile(0.5)

    # --------------------------------------------------------------- density
    def pdf(self, x: float) -> float:
        """Piecewise-constant density at ``x`` (bin_mass / bin_width).
        Returns 0 outside the support."""
        x = float(x)
        if x < self.edges[0] or x >= self.edges[-1]:
            return 0.0
        j = int(np.searchsorted(self.edges, x, side="right") - 1)
        j = max(0, min(j, self.num_bins - 1))
        lo, hi = self.edges[j], self.edges[j + 1]
        return float(self.bin_masses[j]) / (hi - lo) if hi > lo else 0.0

    # ------------------------------------------------------------ visualisation
    def plot(self, ax: Optional[object] = None, **kwargs) -> Optional[object]:
        """Plot the posterior as a histogram.  Matplotlib is optional — if
        not installed, returns ``None`` and does nothing."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            return None
        if ax is None:
            _, ax = plt.subplots()
        widths = np.diff(self.edges)
        heights = self.bin_masses / widths  # density
        ax.bar(self.edges[:-1], heights, width=widths, align="edge",
               edgecolor="k", linewidth=0.5, **kwargs)
        ax.set_xlabel(self.name)
        ax.set_ylabel("density")
        ax.set_title(f"Posterior of {self.name}")
        return ax

    # ------------------------------------------------------------------ dunder
    def __repr__(self) -> str:
        return (
            f"ContinuousPosterior(name='{self.name}', num_bins={self.num_bins}, "
            f"mean={self.mean():.4g}, std={self.std():.4g}, "
            f"support={self.support})"
        )

    def __len__(self) -> int:
        return self.num_bins
