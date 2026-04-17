#!/usr/bin/env python3
"""Sensitivity analysis and Value of Information.

Demonstrates two advanced discrete-BN queries beyond simple marginals:

- **Sensitivity analysis**: rank which CPT entries most influence a target
  query posterior.  Useful for parameter-importance studies.
- **Value of Information (VoI)**: rank which unobserved variables would
  most reduce uncertainty about the query if we could observe them.

Run: python examples/02_sensitivity_and_voi.py
"""
import numpy as np

from pybncore import Graph, PyBNCoreWrapper


def build_diagnostic_model() -> PyBNCoreWrapper:
    """Small medical BN:

        Disease ──▶ Symptom1
              └────▶ Symptom2
    Test1 ──▶ Symptom1
    Test2 ──▶ Symptom2

    Symptoms are noisy observations; tests are additional indicators we
    might choose to perform.
    """
    w = PyBNCoreWrapper()
    g = Graph()
    g.add_variable("Disease", ["Present", "Absent"])
    g.add_variable("Test1", ["Positive", "Negative"])
    g.add_variable("Test2", ["Positive", "Negative"])
    g.add_variable("Symptom1", ["Yes", "No"])
    g.add_variable("Symptom2", ["Yes", "No"])
    g.add_edge("Disease", "Symptom1")
    g.add_edge("Disease", "Symptom2")
    g.add_edge("Test1", "Symptom1")
    g.add_edge("Test2", "Symptom2")

    # Priors and CPTs.  For sensitivity analysis the wrapper requires
    # CPTs to be stored in the wrapper's own `_cpts` dict as well — we
    # set both.
    cpts = {
        "Disease":  np.array([0.02, 0.98]),
        "Test1":    np.array([0.50, 0.50]),
        "Test2":    np.array([0.50, 0.50]),
        # P(Symptom | Disease, Test) — 4 parent configs × 2 states
        # (D=P,T=Pos) (D=P,T=Neg) (D=A,T=Pos) (D=A,T=Neg)
        "Symptom1": np.array([0.90, 0.10,
                              0.70, 0.30,
                              0.40, 0.60,
                              0.05, 0.95]),
        "Symptom2": np.array([0.85, 0.15,
                              0.60, 0.40,
                              0.50, 0.50,
                              0.03, 0.97]),
    }
    for name, cpt in cpts.items():
        g.set_cpt(name, cpt)

    w._graph = g
    w._cpts = dict(cpts)  # sensitivity_ranking() reads from here
    w._cache_metadata()
    return w


def main() -> dict:
    w = build_diagnostic_model()

    # We observe that both symptoms are present.
    w.set_evidence({"Symptom1": "Yes", "Symptom2": "Yes"})

    marginals = w.batch_query_marginals(["Disease"])
    print(f"P(Disease=Present | Symptom1=Yes, Symptom2=Yes) = "
          f"{marginals['Disease']['Present']:.4f}")

    # Sensitivity: rank CPT entries by influence on the query
    print("\nTop 5 sensitive CPT entries for Disease=Present:")
    ranking = w.sensitivity_ranking(
        query_node="Disease", query_state="Present",
    )
    for entry in ranking[:5]:
        print(f"  {entry}")

    # Value of Information: which additional test to perform?
    print("\nValue of Information for candidate tests:")
    voi = w.value_of_information(
        query_node="Disease",
        candidate_nodes=["Test1", "Test2"],
    )
    for node, score in voi:
        print(f"  {node:<6}  VoI = {score:.6f} bits")

    return {
        "disease_given_symptoms": marginals["Disease"]["Present"],
        "voi_ranking": voi,
    }


if __name__ == "__main__":
    main()
