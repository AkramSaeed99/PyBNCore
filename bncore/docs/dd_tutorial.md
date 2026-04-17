# Dynamic Discretization Tutorial

This tutorial walks through PyBNCore's hybrid (continuous-variable) API.
It assumes you've already used the discrete API (see the main README).

## What is dynamic discretization?

Standard junction-tree inference only works on discrete variables. Real
engineering models (reliability, sensor fusion, diagnostics) frequently
involve continuous variables — loads, temperatures, degradation rates,
measurement noise.

Dynamic discretization (DD) bridges this gap by:

1. Representing each continuous variable as a grid of `m` bins
2. Running the normal discrete inference on this grid
3. Measuring per-bin approximation error on the resulting posterior
4. Splitting high-error bins and repeating, until error is below a threshold

PyBNCore implements the **Neil–Tailor–Marquez (2007)** algorithm — the same
one used by the commercial tool AgenaRisk — extended with
**Zhu-Collette (2015)** rare-event reweighting for tail-probability
accuracy.

## Quick start

```python
from pybncore import PyBNCoreWrapper

w = PyBNCoreWrapper()
w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0))

result = w.hybrid_query(["X"])
posterior = result["X"]

print(f"mean = {posterior.mean():.3f}")          # ~0.0
print(f"std  = {posterior.std():.3f}")           # ~1.0
print(f"P(X < -1) = {posterior.prob_less_than(-1):.3f}")  # ~0.159
print(f"90th %ile = {posterior.quantile(0.9):.3f}")       # ~1.28
```

The returned object is a `ContinuousPosterior` — see
[API reference](#the-continuousposterior-api) below.

## Distribution families

PyBNCore supports five built-in distribution families, plus deterministic
functional nodes and a user-supplied integrator for custom densities.

### Normal

```python
w.add_normal("X", mu=0.0, sigma=1.0, domain=(-4.0, 4.0))
```

Parameters `mu` and `sigma` may be scalars or callables taking parent
values. For example, a Markov chain step:

```python
w.add_normal("X1", parents=["X0"],
             mu=lambda x0: x0,              # identity transition
             sigma=0.1,                      # low noise
             domain=(-4.0, 4.0))
```

### LogNormal

Used extensively in reliability analysis (degradation rates, time-to-failure):

```python
w.add_lognormal(
    "R", log_mu=-2.0, log_sigma=0.5,
    domain=(1e-4, 10.0),        # must be strictly positive
    log_spaced=True,             # log-uniform initial grid
)
```

### Uniform, Exponential

```python
w.add_uniform("C", a=8.0, b=12.0, domain=(8.0, 12.0))
w.add_exponential("T", rate=0.5, domain=(0.0, 20.0))
```

### Deterministic

For computed variables, `Y = fn(parents)`. Example: total stress as the
product of degradation and load divided by capacity.

```python
w.add_deterministic(
    "S", parents=["R", "L", "C"],
    fn=lambda r, l, c: r * l / c,
    domain=(0.0, 5.0),
    monotone=False,  # pass True for guaranteed-monotone `fn` — exact
)
```

For monotone functions (e.g. `Y = log(X)`), set `monotone=True` for an
exact interval-arithmetic distribution. Non-monotone functions use Monte
Carlo sampling (32 samples by default).

## Adding evidence by value

Hard evidence on a continuous variable — pin it to an observed value:

```python
w.set_continuous_evidence({"X0": 1.2, "X1": -0.5})
```

The value is re-resolved to a bin index at every iteration, so it
remains valid even after refinement splits the containing bin.

Discrete and continuous evidence can coexist:

```python
w.set_evidence({"DiscreteNode": "True"})          # normal discrete API
w.set_continuous_evidence({"Temp": 25.0})          # continuous value
```

### Soft evidence as a likelihood function

For uncertain continuous observations — supply a likelihood density
`λ(x)`:

```python
import math
# Gaussian measurement noise around observed = 0.5, std = 0.05
w.set_continuous_likelihood(
    "Sensor",
    lambda x: math.exp(-0.5 * ((x - 0.5) / 0.05) ** 2)
)
```

The likelihood is integrated over each bin at every iteration.

## Rare-event analysis

The default entropy-error refinement puts resolution where the density
is tallest — usually the peak, not the tails. For PRA / reliability
queries of the form `P(X > threshold)` where the probability is small,
this is wrong. Use **rare-event mode** and **threshold seeding** together:

```python
w.add_lognormal(
    "R", log_mu=-2.0, log_sigma=0.5,
    domain=(1e-4, 5.0),
    log_spaced=True,
    rare_event_mode=True,       # Zhu-Collette reweighting
)
w.add_threshold("R", 0.05)      # always keep an edge at 0.05

result = w.hybrid_query(["R"])
p_failure = result["R"].prob_less_than(0.05)
# Exact to 4+ decimal places regardless of iteration count
```

**Why this works:**

- `rare_event_mode=True` divides each bin's entropy error by its posterior
  mass, so low-mass tail bins get prioritised for splitting.
- `add_threshold(node, value)` guarantees there's always a bin edge
  exactly at `value`, so the tail-probability integral `P(X < value)`
  has a clean boundary instead of being interpolated.

Without these, tail probabilities can be off by a factor of 10 at the
first iteration and take many iterations to converge. With them, tail
probabilities are often exact on iteration 1.

## Deterministic / functional nodes

For computed variables like `Y = g(X1, X2)`:

```python
# Y = X1 + X2 (non-monotone in each, but Monte-Carlo works)
w.add_deterministic(
    "Y", parents=["X1", "X2"],
    fn=lambda x1, x2: x1 + x2,
    domain=(-8.0, 8.0),
    n_samples=64,                 # increase for more accuracy
)

# Y = log(X) (monotone — use exact interval arithmetic)
w.add_deterministic(
    "Y", parents=["X"],
    fn=lambda x: math.log(x),
    domain=(-5.0, 5.0),
    monotone=True,
)
```

When `monotone=True` is set but `fn` isn't actually monotone, mass may
leak; only set it when you're sure.

## The ContinuousPosterior API

`hybrid_query` returns a `HybridResult` indexed by variable name.
Continuous variables produce `ContinuousPosterior` objects:

```python
p = result["X"]
# Summary statistics
p.mean(), p.variance(), p.std(), p.median()
p.mode_bin()                           # argmax bin index

# Probability queries
p.cdf(1.5)                             # P(X < 1.5)
p.prob_less_than(1.5)                  # same
p.prob_greater_than(1.5)
p.prob_between(0.0, 1.0)               # P(0 <= X < 1)
p.quantile(0.95)                       # 95th percentile

# Density evaluation
p.pdf(0.5)                              # density at x = 0.5

# Plotting (optional, matplotlib)
p.plot()

# Raw access
p.edges                                 # np.ndarray shape (m+1,)
p.bin_masses                           # np.ndarray shape (m,)
p.num_bins
p.support                              # (domain_lo, domain_hi)
```

Discrete query variables still return the familiar dict:

```python
result["DiscreteNode"]   # {"State1": 0.7, "State2": 0.3}
```

## Convergence diagnostics

```python
result = w.hybrid_query(["X"], max_iters=10, eps_entropy=1e-4)

print(result.converged)        # True/False
print(result.iterations_used)  # How many outer iterations ran
print(result.final_max_error)  # Largest per-bin entropy error remaining
```

If `converged=False` after `max_iters`, you have three options:

1. Increase `max_iters` (default 8; try 20)
2. Loosen `eps_entropy` (default 1e-4)
3. Increase `initial_bins` to start closer to the solution

For tail-probability queries, `rare_event_mode=True` + `add_threshold`
usually converges in 2–3 iterations.

## Troubleshooting

**"initialize_graph: variable 'R' has 12 states in the graph but 13 bins
registered"**

You called `add_threshold` after building the graph in a way that caused
a mismatch. The wrapper handles this automatically via `initialize_graph`,
which grows the graph state list. If you see this error at the C++ level,
call `hybrid_query` (which resyncs) rather than raw `_core` APIs.

**Poor tail accuracy**

Almost always means you forgot `rare_event_mode=True` and/or
`add_threshold(node, value)`. Both are required to get reliable
tail-probability answers.

**Posterior doesn't look Gaussian**

Check that `domain` is wide enough to cover the tails. For a
Normal(0, 1), domain of `(-4, 4)` captures 99.99%. For a LogNormal or
Exponential, use a log-spaced grid (`log_spaced=True`).

**"batch_query_marginals requires hybrid_query for hybrid models"**

You registered at least one continuous variable, which puts the wrapper
in hybrid mode. Use `wrapper.hybrid_query(...)` instead of
`batch_query_marginals(...)`.

## See also

- [Flagship example — reliability analysis](../examples/04_reliability_rare_event.py)
- [Gaussian chain example](../examples/03_hybrid_gaussian_chain.py)
- [Rare-event paper: Zhu & Collette, 2015](https://www.sciencedirect.com/science/article/abs/pii/S0951832015000277)
- [NTM algorithm paper: Neil, Tailor & Marquez, 2007](https://link.springer.com/article/10.1007/s11222-007-9018-y)

## Known limitations (planned)

- **Batched DD:** currently each `hybrid_query` processes one evidence
  row. Monte Carlo / UQ workflows over thousands of parameter samples
  could share the same bin grid — planned.
- **Continuous-var integration with sensitivity() and VoI():** the
  advanced discrete-BN analyses don't yet operate on hybrid models.
