#!/usr/bin/env python3
"""Hybrid BN: Gaussian chain with dynamic discretization.

Model:
    X0 ~ Normal(0, 1)
    X1 | X0 ~ Normal(X0, 0.3)      (low-noise transition)
    X2 | X1 ~ Normal(X1, 0.3)

This is the Kalman-filter intuition: each node's posterior mean tracks
its parent's observation.  Closed-form: if we observe X0 = x*, then
E[X1 | X0=x*] = x* exactly, and the marginal variance builds up along
the chain.

Run: python examples/03_hybrid_gaussian_chain.py
"""
from pybncore import PyBNCoreWrapper


def build_model() -> PyBNCoreWrapper:
    w = PyBNCoreWrapper()
    # All three variables are continuous Gaussians.  Parents are passed
    # by name; the callback receives the parent's midpoint value.
    w.add_normal("X0", mu=0.0, sigma=1.0, domain=(-4.0, 4.0), initial_bins=10)
    w.add_normal("X1", parents=["X0"],
                  mu=lambda x0: x0, sigma=0.3,
                  domain=(-4.0, 4.0), initial_bins=10)
    w.add_normal("X2", parents=["X1"],
                  mu=lambda x1: x1, sigma=0.3,
                  domain=(-4.0, 4.0), initial_bins=10)
    return w


def main() -> dict:
    w = build_model()
    observed_x0 = 1.2

    print(f"Setting evidence X0 = {observed_x0}")
    w.set_continuous_evidence({"X0": observed_x0})

    result = w.hybrid_query(["X0", "X1", "X2"], max_iters=6)
    print(f"\nConvergence: {result}")
    print()

    print(f"{'Variable':<8} {'mean':>9} {'std':>9} {'median':>9} "
          f"{'P(X>0)':>9}")
    print("  " + "-" * 50)
    summary = {}
    for name in ["X0", "X1", "X2"]:
        p = result[name]
        row = {
            "mean": p.mean(),
            "std":  p.std(),
            "median": p.median(),
            "p_gt_0": p.prob_greater_than(0.0),
        }
        summary[name] = row
        print(f"  {name:<6} {row['mean']:>9.3f} {row['std']:>9.3f} "
              f"{row['median']:>9.3f} {row['p_gt_0']:>9.3f}")

    print(
        f"\nExpected (closed-form, infinite bins):"
        f"\n  X0 ≡ {observed_x0} (pinned)"
        f"\n  X1 ~ N({observed_x0}, 0.3)"
        f"\n  X2 ~ N({observed_x0}, sqrt(0.3² + 0.3²)) = N({observed_x0}, 0.424)"
    )

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
        for ax, name in zip(axes, ["X0", "X1", "X2"]):
            result[name].plot(ax=ax)
        fig.suptitle(f"Gaussian chain posterior (X0 observed at {observed_x0})")
        fig.tight_layout()
        fig.savefig("gaussian_chain.png", dpi=120)
        print("\nPlot saved to gaussian_chain.png")
    except ImportError:
        pass

    return summary


if __name__ == "__main__":
    main()
