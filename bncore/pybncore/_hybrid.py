"""Internal helpers for the hybrid (continuous variable) API.

These are implementation details of :class:`pybncore.wrapper.PyBNCoreWrapper`
and are **not** part of the public API.  Everything here operates directly
on the raw ``pybncore._core`` bindings; the wrapper is responsible for
keeping its own state in sync.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple, Union


# ----------------------------------------------------------------------------
# Parameter kinds
# ----------------------------------------------------------------------------
# A distribution parameter (e.g. `mu` for a Normal) may be either:
#   * a fixed scalar (e.g. `mu=0.0`)
#   * a callable that takes the parent values, in declared order, and returns
#     a float (e.g. `mu=lambda x0: x0`)
# These aliases make the wrapper signatures self-documenting.
Scalar = Union[int, float]
ParamFn = Callable[..., float]
Param = Union[Scalar, ParamFn]


def _wrap_param(param: Param, parent_names: Sequence[str],
                 continuous_names: Sequence[str]):
    """Convert a user-supplied scalar or callable into a C++ ``ParentBins``
    consumer.

    The raw C++ callback signature is ``fn(pb: ParentBins) -> float`` where
    ``pb.continuous_values`` and ``pb.discrete_states`` are zipped in the
    order the parents were declared (continuous first in their list, discrete
    next in theirs).  Users write their callbacks in the natural order of
    ``parents``; this function reorders arguments to match.

    Parameters
    ----------
    param :
        Either a scalar (returned as-is for every ParentBins) or a callable.
    parent_names :
        The full ordered list of this variable's parent names, as declared.
    continuous_names :
        Subset of parent_names that are continuous variables.

    Returns
    -------
    callable
        A one-argument function suitable for passing to the C++ layer.
    """
    # Scalars short-circuit: wrap as constant lambda.
    if not callable(param):
        value = float(param)
        return lambda pb, _v=value: _v

    # Precompute mapping: for each declared parent, record whether it's
    # continuous and its index into the corresponding ParentBins array.
    cont_set = set(continuous_names)
    # Indices within pb.continuous_values / pb.discrete_states are assigned
    # in the order declared parents appear, split by kind.  The C++ side
    # mirrors this: see DiscretizationManager::rebuild_cpts which builds
    # ParentBins by iterating parents in their declared order and pushing
    # to whichever vector is appropriate.
    kind_and_index: List[Tuple[str, int]] = []
    ci = 0
    di = 0
    for p in parent_names:
        if p in cont_set:
            kind_and_index.append(("cont", ci))
            ci += 1
        else:
            kind_and_index.append(("disc", di))
            di += 1

    def wrapped(pb, _param=param, _kai=kind_and_index):
        args = []
        for kind, idx in _kai:
            if kind == "cont":
                args.append(pb.continuous_values[idx])
            else:
                args.append(pb.discrete_states[idx])
        return float(_param(*args))

    return wrapped


# ----------------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------------
def _as_float(v: Any, ctx: str) -> float:
    """Coerce to float or raise TypeError with a message that says *where*
    in the user's call the bad value came from."""
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{ctx} must be a number, got {type(v).__name__}: {v!r}"
        ) from exc


def _validate_domain(domain: Tuple[float, float], ctx: str) -> Tuple[float, float]:
    if not (isinstance(domain, tuple) and len(domain) == 2):
        raise ValueError(
            f"{ctx}: domain must be a 2-tuple (lo, hi), got {domain!r}"
        )
    lo, hi = _as_float(domain[0], f"{ctx}.domain[0]"), \
              _as_float(domain[1], f"{ctx}.domain[1]")
    if hi <= lo:
        raise ValueError(
            f"{ctx}: domain upper bound ({hi}) must be > lower bound ({lo})"
        )
    return lo, hi


def _validate_initial_bins(initial_bins: int, ctx: str) -> int:
    if not isinstance(initial_bins, int) or initial_bins < 2:
        raise ValueError(
            f"{ctx}: initial_bins must be an int >= 2, got {initial_bins!r}"
        )
    return int(initial_bins)
