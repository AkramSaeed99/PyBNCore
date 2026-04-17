#!/usr/bin/env python3
"""Diagnostic reasoning with soft (virtual) evidence.

Soft evidence represents uncertain observations, such as a sensor reading
with known noise.  Instead of clamping a variable to a specific state,
you supply a likelihood vector proportional to the measurement's
likelihood under each underlying state.

Model:
    Condition ──▶ TestResult

Imagine the test is 85% accurate when positive and 70% accurate when
negative — i.e. an imperfect observation.  Standard hard evidence would
over-commit.  We model this using a likelihood ratio.

Run: python examples/05_diagnostic_with_soft_evidence.py
"""
import numpy as np

from pybncore import Graph, PyBNCoreWrapper


def build_model() -> PyBNCoreWrapper:
    w = PyBNCoreWrapper()
    g = Graph()
    g.add_variable("Condition", ["Has", "NoHas"])
    g.add_variable("TestResult", ["Positive", "Negative"])
    g.add_edge("Condition", "TestResult")

    # Base rate: rare condition.
    g.set_cpt("Condition", np.array([0.03, 0.97]))
    # P(TestResult | Condition)
    #   Sensitivity: P(Pos | Has)   = 0.95
    #   Specificity: P(Neg | NoHas) = 0.92
    g.set_cpt("TestResult", np.array([0.95, 0.05,
                                       0.08, 0.92]))

    w._graph = g
    w._cache_metadata()
    return w


def main() -> dict:
    w = build_model()

    # 1. Prior probability of the condition
    prior = w.batch_query_marginals(["Condition"])["Condition"]
    print(f"Prior          P(Condition=Has) = {prior['Has']:.4f}")

    # 2. Hard positive test: standard evidence
    w.set_evidence({"TestResult": "Positive"})
    post_hard = w.batch_query_marginals(["Condition"])["Condition"]
    print(f"Hard Positive  P(Condition=Has) = {post_hard['Has']:.4f}")

    # 3. Soft positive test: we "believe" it's positive, likelihood ratio 3:1
    w.clear_evidence()
    w.set_soft_evidence("TestResult", {"Positive": 0.75, "Negative": 0.25})
    post_soft = w.batch_query_marginals(["Condition"])["Condition"]
    print(f"Soft Positive  P(Condition=Has) = {post_soft['Has']:.4f}")

    # 4. The test is "inconclusive": 50/50 soft evidence → same as prior
    w.clear_soft_evidence()
    w.set_soft_evidence("TestResult", {"Positive": 0.5, "Negative": 0.5})
    post_incon = w.batch_query_marginals(["Condition"])["Condition"]
    print(f"Inconclusive   P(Condition=Has) = {post_incon['Has']:.4f}")
    print(f"(Matches prior: {prior['Has']:.4f} ✓)")

    return {
        "prior":        prior["Has"],
        "hard_positive": post_hard["Has"],
        "soft_positive": post_soft["Has"],
        "inconclusive":  post_incon["Has"],
    }


if __name__ == "__main__":
    main()
