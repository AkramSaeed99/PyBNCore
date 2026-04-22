"""End-to-end service-layer tests against the bundled fixture XDSL."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pybncore")

from pybncore_gui.domain.continuous import ContinuousDistKind, ContinuousNodeSpec
from pybncore_gui.domain.errors import EvidenceError, QueryError
from pybncore_gui.domain.session import ModelSession
from pybncore_gui.services.analysis_service import AnalysisService
from pybncore_gui.services.authoring_service import AuthoringService
from pybncore_gui.services.inference_service import InferenceService
from pybncore_gui.services.io_service import IOService
from pybncore_gui.services.model_service import ModelService
from pybncore_gui.services.submodel_service import SubModelService
from pybncore_gui.services.validation_service import ValidationService


@pytest.fixture
def session_with_asia(asia_mini_path: pathlib.Path):
    session = ModelSession()
    io = IOService(session)
    io.open_xdsl(asia_mini_path)
    return session


# --------------------------------------------------------------- model


def test_open_xdsl_lists_structure(session_with_asia):
    ms = ModelService(session_with_asia)
    ids = {n.id for n in ms.list_nodes()}
    assert ids == {"Smoker", "Cancer", "XRay"}
    edges = {(e.parent, e.child) for e in ms.list_edges()}
    assert edges == {("Smoker", "Cancer"), ("Cancer", "XRay")}
    assert ms.get_outcomes("Smoker") == ["yes", "no"]


# ---------------------------------------------------------- authoring


def test_add_and_remove_discrete_node(session_with_asia):
    auth = AuthoringService(session_with_asia)
    ms = ModelService(session_with_asia)
    auth.add_discrete_node("Dyspnea", ["yes", "no"], parents=["Cancer"])
    ids = {n.id for n in ms.list_nodes()}
    assert "Dyspnea" in ids
    snapshot = auth.remove_node("Dyspnea")
    assert snapshot.name == "Dyspnea"
    assert "Dyspnea" not in {n.id for n in ms.list_nodes()}


def test_rename_node_preserves_edges(session_with_asia):
    auth = AuthoringService(session_with_asia)
    ms = ModelService(session_with_asia)
    auth.rename_node("Cancer", "LungCancer")
    ids = {n.id for n in ms.list_nodes()}
    assert "Cancer" not in ids and "LungCancer" in ids
    edges = {(e.parent, e.child) for e in ms.list_edges()}
    assert ("Smoker", "LungCancer") in edges
    assert ("LungCancer", "XRay") in edges


def test_edit_states_resets_child_cpt(session_with_asia):
    auth = AuthoringService(session_with_asia)
    ms = ModelService(session_with_asia)
    # Cancer had two states; bump to three.
    before = auth.node_snapshot("Cancer")
    auth.update_node_states("Cancer", ("none", "mild", "severe"))
    updated = {n.id: n for n in ms.list_nodes()}
    assert updated["Cancer"].states == ("none", "mild", "severe")
    # XRay's CPT must match new cardinalities (3 parent states × 2 child).
    cpt = ms.get_cpt_shaped("XRay").reshape(-1, 2)
    assert cpt.shape == (3, 2)
    assert np.allclose(cpt.sum(axis=1), 1.0)
    # Sanity: restoring via update restores back to 2 states.
    auth.update_node_states("Cancer", before.states)
    again = ms.get_cpt_shaped("XRay").reshape(-1, 2)
    assert again.shape == (2, 2)


def test_cycle_is_rejected_on_add_edge(session_with_asia):
    auth = AuthoringService(session_with_asia)
    # Smoker → Cancer → XRay already exists; adding XRay → Smoker closes a cycle.
    with pytest.raises(EvidenceError):
        auth.add_edge("XRay", "Smoker")


# ---------------------------------------------------------- inference


def test_compile_and_single_query(session_with_asia):
    inf = InferenceService(session_with_asia)
    stats = inf.compile()
    assert stats.node_count == 3
    # Prior posterior for Smoker — uniform in the fixture.
    posterior = inf.query_single("Smoker", evidence={})
    assert posterior.states == ("yes", "no")
    assert pytest.approx(sum(posterior.probabilities), rel=1e-6) == 1.0
    assert pytest.approx(posterior.probabilities[0], rel=1e-3) == 0.5

    # With Cancer observed positive, Smoker should shift toward yes.
    conditional = inf.query_single("Smoker", evidence={"Cancer": "yes"})
    assert conditional.probabilities[0] > posterior.probabilities[0]


def test_map_returns_assignment(session_with_asia):
    inf = InferenceService(session_with_asia)
    result = inf.query_map(evidence={"XRay": "pos"})
    assert set(result.assignment.keys()) == {"Smoker", "Cancer", "XRay"}
    # XRay pos observation is respected.
    assert result.assignment["XRay"] == "pos"


def test_jt_stats_after_compile(session_with_asia):
    inf = InferenceService(session_with_asia)
    inf.compile()
    stats = inf.compute_jt_stats()
    assert stats.num_cliques >= 1
    assert stats.treewidth >= 1


# -------------------------------------------------------------- analysis


def test_sensitivity_ranking(session_with_asia):
    analysis = AnalysisService(session_with_asia)
    report = analysis.sensitivity_ranking(
        "Cancer", "yes", n_top=5, epsilon=0.05
    )
    assert report.entries, "expected at least one sensitivity entry"
    assert all(isinstance(e.score, float) for e in report.entries)


def test_value_of_information(session_with_asia):
    analysis = AnalysisService(session_with_asia)
    report = analysis.value_of_information("Cancer")
    names = [e.candidate for e in report.entries]
    assert "Smoker" in names or "XRay" in names


def test_benchmark_runs(session_with_asia):
    analysis = AnalysisService(session_with_asia)
    result = analysis.benchmark(
        query_nodes=["Cancer"],
        observed_nodes=["XRay"],
        row_counts=[4, 8],
        seed=0,
    )
    assert len(result.points) == 2
    assert all(pt.elapsed_ms >= 0 for pt in result.points)


def test_monte_carlo_summaries(session_with_asia):
    analysis = AnalysisService(session_with_asia)
    result = analysis.monte_carlo(
        query_nodes=["Cancer"],
        observed_nodes=["Smoker"],
        num_samples=32,
        seed=1,
    )
    summary = result.summaries["Cancer"]
    assert pytest.approx(sum(summary.mean), rel=1e-3) == 1.0


# -------------------------------------------------------------- validation


def test_validation_clean_model(session_with_asia):
    val = ValidationService(session_with_asia)
    report = val.validate()
    assert not report.has_errors


def test_validation_flags_bad_cpt(session_with_asia):
    auth = AuthoringService(session_with_asia)
    val = ValidationService(session_with_asia)
    # Write a non-stochastic CPT into Smoker.
    with session_with_asia.locked() as wrapper:
        wrapper._cpts["Smoker"] = np.array([0.2, 0.3], dtype=np.float64)
        wrapper._graph.set_cpt("Smoker", np.array([0.2, 0.3], dtype=np.float64))
    report = val.validate()
    assert report.has_errors
    assert any(i.code == "cpt_rows" for i in report.issues)


# --------------------------------------------------------------- sub-models


def test_submodel_round_trip(asia_mini_path: pathlib.Path, tmp_path):
    svc = SubModelService()
    layout = svc.parse_from_xdsl(asia_mini_path)
    assert set(layout.submodels.keys()) == {"SUBMODEL_RISK"}
    assert layout.node_parent["Smoker"] == "SUBMODEL_RISK"
    assert layout.node_parent["Cancer"] == "SUBMODEL_RISK"
    assert layout.node_parent["XRay"] == ""
    desc = svc.parse_descriptions(asia_mini_path)
    assert desc["XRay"].startswith("Chest x-ray finding")


# -------------------------------------------------------------- continuous


def test_continuous_creation_and_hybrid_query(tmp_path):
    """Fresh hybrid model exercised end-to-end."""
    from pybncore_gui.domain.errors import EvidenceError as _EE  # noqa: F401
    session = ModelSession()
    io = IOService(session)
    io.new_empty()
    auth = AuthoringService(session)
    spec = ContinuousNodeSpec(
        name="X",
        kind=ContinuousDistKind.NORMAL,
        parents=(),
        domain=(-5.0, 5.0),
        initial_bins=8,
        params={"mu": 0.0, "sigma": 1.0},
    )
    auth.add_continuous_node(spec)
    inf = InferenceService(session)
    result = inf.run_hybrid(["X"], max_iters=3)
    assert "X" in result.continuous
    posterior = result.continuous["X"]
    assert posterior.num_bins >= 8
    assert posterior.support[0] < 0 < posterior.support[1]
    # PDF and CDF grids are populated.
    assert len(posterior.pdf_grid) > 16
    assert posterior.cdf_grid[-1][1] > posterior.cdf_grid[0][1]
