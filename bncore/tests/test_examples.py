"""Smoke tests: every example in `examples/` must run to completion and
return sensible results.  This catches silent doc-code drift.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest


EXAMPLES = [
    "01_hello_discrete",
    "02_sensitivity_and_voi",
    "03_hybrid_gaussian_chain",
    "04_reliability_rare_event",
    "05_diagnostic_with_soft_evidence",
]


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _import_example(name: str):
    """Import a file from `examples/` as a module without polluting sys.path."""
    path = EXAMPLES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Example {name}.py not found")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name, tmp_path, monkeypatch):
    """Each example's `main()` runs without exception."""
    # Run in a temp dir so any `plt.savefig("...png")` calls don't pollute the repo
    monkeypatch.chdir(tmp_path)
    mod = _import_example(name)
    assert hasattr(mod, "main"), f"{name} must expose main()"
    result = mod.main()
    assert result is not None, f"{name}.main() must return a dict"


# ---------------------------------------------------------------------------
# Specific invariants per example
# ---------------------------------------------------------------------------
def test_01_hello_discrete_sensible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _import_example("01_hello_discrete").main()
    # Classic rain/sprinkler: wet grass is more likely than not when it rains
    assert 0.0 < r["prior_wetgrass_true"] < 1.0
    # Observing wet grass should INCREASE P(Rain) above prior 0.2
    assert r["posterior_rain_true"] > 0.2


def test_02_sensitivity_returns_valid_voi(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _import_example("02_sensitivity_and_voi").main()
    # VoI values must all be non-negative
    for _, score in r["voi_ranking"]:
        assert score >= 0, "VoI scores must be non-negative"


def test_03_gaussian_chain_tracks_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _import_example("03_hybrid_gaussian_chain").main()
    # Observed X0 = 1.2; descendants should have posterior mean close to 1.2
    assert abs(r["X0"]["mean"] - 1.2) < 0.15
    assert abs(r["X1"]["mean"] - 1.2) < 0.2
    assert abs(r["X2"]["mean"] - 1.2) < 0.3


def test_04_reliability_returns_valid_probability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _import_example("04_reliability_rare_event").main()
    for k in ("p_failure_default", "p_failure_rare_event"):
        assert 0.0 <= r[k] <= 1.0, (
            f"{k} must be a valid probability, got {r[k]}"
        )
    # Stress is bounded; 95th percentile must lie in the domain
    assert r["stress_95pct"] > 0.0


def test_05_soft_evidence_between_prior_and_hard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _import_example("05_diagnostic_with_soft_evidence").main()
    # Weak-positive soft evidence should lift posterior above prior but
    # below the hard-positive result.
    assert r["prior"] < r["soft_positive"] < r["hard_positive"]
    # 50/50 soft evidence must reproduce the prior
    assert abs(r["inconclusive"] - r["prior"]) < 1e-9
