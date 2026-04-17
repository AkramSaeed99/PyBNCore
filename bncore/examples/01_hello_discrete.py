#!/usr/bin/env python3
"""Hello-world discrete Bayesian network: Rain / Sprinkler / WetGrass.

Demonstrates the most basic PyBNCore usage:
- Build a DAG
- Assign conditional probability tables (CPTs)
- Query the posterior given evidence

No continuous variables, no dynamic discretization — just the classic
textbook example.  Run this first if you're new to the library.

Run: python examples/01_hello_discrete.py
"""
import numpy as np

from pybncore import Graph, PyBNCoreWrapper


def build_model() -> PyBNCoreWrapper:
    """Classic rain/sprinkler/wet-grass model.

    DAG:   Rain ──┐
                   ├──▶ WetGrass
         Sprinkler ┘

    Rain flips the probability of Sprinkler being on (you don't water if it's raining).
    """
    w = PyBNCoreWrapper()
    g = Graph()
    g.add_variable("Rain", ["True", "False"])
    g.add_variable("Sprinkler", ["On", "Off"])
    g.add_variable("WetGrass", ["True", "False"])
    g.add_edge("Rain", "Sprinkler")
    g.add_edge("Rain", "WetGrass")
    g.add_edge("Sprinkler", "WetGrass")

    # CPTs (row-major, child state is innermost axis):
    # P(Rain):                                   [True, False]
    g.set_cpt("Rain",      np.array([0.2, 0.8]))
    # P(Sprinkler | Rain):    Rain=T → [On, Off],  Rain=F → [On, Off]
    g.set_cpt("Sprinkler", np.array([0.01, 0.99,
                                      0.4,  0.6]))
    # P(WetGrass | Rain, Sprinkler):
    #   (R=T, S=On) → [T, F]
    #   (R=T, S=Off) → [T, F]
    #   (R=F, S=On) → [T, F]
    #   (R=F, S=Off) → [T, F]
    g.set_cpt("WetGrass",  np.array([0.99, 0.01,
                                      0.80, 0.20,
                                      0.90, 0.10,
                                      0.00, 1.00]))

    w._graph = g
    w._cache_metadata()
    return w


def main() -> dict:
    w = build_model()

    # Prior marginal for WetGrass (no evidence)
    prior = w.batch_query_marginals(["WetGrass"])
    print("Prior  P(WetGrass=True) = "
          f"{prior['WetGrass']['True']:.4f}")

    # Posterior after observing wet grass
    w.set_evidence({"WetGrass": "True"})
    post = w.batch_query_marginals(["Rain", "Sprinkler"])
    print(f"Given WetGrass=True:")
    print(f"  P(Rain=True)      = {post['Rain']['True']:.4f}")
    print(f"  P(Sprinkler=On)   = {post['Sprinkler']['On']:.4f}")

    return {
        "prior_wetgrass_true": prior["WetGrass"]["True"],
        "posterior_rain_true": post["Rain"]["True"],
        "posterior_sprinkler_on": post["Sprinkler"]["On"],
    }


if __name__ == "__main__":
    main()
