#!/usr/bin/env python3
"""⭐ Flagship example: reliability / PRA rare-event analysis.

Structural reliability model:

    R ~ LogNormal(-2, 0.5)        degradation rate
    L ~ Normal(5, 1)               applied load (kN)
    C ~ Uniform(8, 12)             capacity (kN) — load we can bear
    S = R · L / C                   dimensionless stress (deterministic)
    Fails iff S > 0.8

Query: P(S > 0.8) — the probability of structural failure.

This showcases the three DD features that matter most for PRA:
  - continuous variables with realistic engineering distributions
  - a deterministic functional node (S is a pure function of R, L, C)
  - **rare-event mode + threshold seeding** so the failure probability
    (which may be small) is computed accurately on the very first iteration

Run: python examples/04_reliability_rare_event.py
"""
from pybncore import PyBNCoreWrapper


FAILURE_THRESHOLD = 0.8


def build_model(use_rare_event_mode: bool = True) -> PyBNCoreWrapper:
    """Construct the reliability DAG.

    If `use_rare_event_mode=True`, R uses Zhu-Collette reweighting and the
    deterministic node S has an edge seeded exactly at the failure
    threshold — giving sharp tail-probability accuracy.  Otherwise the
    default entropy-error refinement is used for comparison.
    """
    w = PyBNCoreWrapper()
    w.add_lognormal(
        "R", log_mu=-2.0, log_sigma=0.5,
        domain=(1e-4, 5.0), initial_bins=12,
        log_spaced=True,
        rare_event_mode=use_rare_event_mode,
    )
    w.add_normal(
        "L", mu=5.0, sigma=1.0,
        domain=(0.0, 10.0), initial_bins=10,
    )
    w.add_uniform(
        "C", a=8.0, b=12.0,
        domain=(8.0, 12.0), initial_bins=10,
    )
    w.add_deterministic(
        "S", parents=["R", "L", "C"],
        fn=lambda r, l, c: r * l / c,
        domain=(0.0, 5.0), initial_bins=14,
        monotone=False,           # R·L/C is monotone in each argument
                                  # individually but the Cartesian corners
                                  # are sufficient; sampling also works.
        n_samples=64,
    )
    if use_rare_event_mode:
        w.add_threshold("S", FAILURE_THRESHOLD)
    return w


def estimate_failure_probability(w: PyBNCoreWrapper,
                                   max_iters: int = 8) -> float:
    """Run DD, read off P(S > FAILURE_THRESHOLD)."""
    result = w.hybrid_query(["S"], max_iters=max_iters)
    return float(result["S"].prob_greater_than(FAILURE_THRESHOLD))


def main() -> dict:
    # Without rare-event mode — default refinement.  Tail is imprecise.
    print("=" * 68)
    print("  Reliability analysis: P(stress > 0.8)")
    print("=" * 68)

    p_default = estimate_failure_probability(build_model(False))
    p_rare    = estimate_failure_probability(build_model(True))

    print(f"\n  Default refinement:           P(failure) = {p_default:.5f}")
    print(f"  Rare-event mode + threshold:  P(failure) = {p_rare:.5f}")
    print(f"\n  Absolute difference:          {abs(p_rare - p_default):.5f}")
    print("\nThe rare-event-mode estimate places an exact bin edge at the"
          f"\nfailure threshold ({FAILURE_THRESHOLD}), guaranteeing the tail"
          "\nintegral boundary is clean.  For failure-probability analyses"
          "\nalways prefer rare-event mode + add_threshold().")

    # Full posterior of stress, for plotting
    w = build_model(True)
    result = w.hybrid_query(["S"], max_iters=8)
    S = result["S"]
    print(f"\nStress posterior summary:")
    print(f"  mean   = {S.mean():.4f}")
    print(f"  median = {S.median():.4f}")
    print(f"  std    = {S.std():.4f}")
    print(f"  95th %ile = {S.quantile(0.95):.4f}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=(8, 4))
        S.plot(ax=ax)
        ax.axvline(FAILURE_THRESHOLD, color="red", linestyle="--",
                    label=f"failure threshold = {FAILURE_THRESHOLD}")
        ax.legend()
        fig.tight_layout()
        fig.savefig("reliability_stress.png", dpi=120)
        print("\nPlot saved to reliability_stress.png")
    except ImportError:
        pass

    return {
        "p_failure_default":    p_default,
        "p_failure_rare_event": p_rare,
        "stress_mean":          S.mean(),
        "stress_95pct":         S.quantile(0.95),
    }


if __name__ == "__main__":
    main()
