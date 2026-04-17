"""PyBNCore — high-performance Bayesian Network inference in C++/Python."""
from ._core import (
    BatchExecutionEngine,
    DiscretizationManager,
    Graph,
    HybridEngine,
    HybridRunConfig,
    JunctionTree,
    JunctionTreeCompiler,
    VariableMetadata,
)
from .io import read_xdsl
from .posterior import ContinuousPosterior
from .wrapper import PyBNCoreWrapper

__all__ = [
    # High-level API (recommended)
    "PyBNCoreWrapper",
    "ContinuousPosterior",
    "read_xdsl",
    # Low-level C++ bindings (advanced users)
    "Graph",
    "VariableMetadata",
    "JunctionTree",
    "JunctionTreeCompiler",
    "BatchExecutionEngine",
    "DiscretizationManager",
    "HybridEngine",
    "HybridRunConfig",
]
