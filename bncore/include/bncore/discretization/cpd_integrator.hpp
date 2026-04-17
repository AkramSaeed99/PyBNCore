#pragma once
// ============================================================================
//  CpdIntegrator — closed-form analytic integrators for continuous CPDs.
//
//  Each integrator computes   P(X in [lo, hi) | parents)   where the parent
//  configuration is summarised either as a representative value (continuous
//  parent's bin midpoint) or as a discrete state index (discrete parent).
//
//  Used by DiscretizationManager::rebuild_cpts() to fill a continuous
//  variable's discrete CPT after a bin-grid change.
// ============================================================================
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace bncore {

// Parent summary passed to an integrator for a single parent configuration.
// Continuous parents contribute a representative value (bin midpoint);
// discrete parents contribute a state index.  The two vectors are
// zipped with the parent order declared when the integrator is registered.
struct ParentBins {
  std::vector<double> continuous_values;  // one per continuous parent
  std::vector<int>    discrete_states;    // one per discrete parent
};

class CpdIntegrator {
public:
  virtual ~CpdIntegrator() = default;

  // Return P(X in [lo, hi) | parents).  Must be in [0, 1] and consistent
  // across bins so that integrating over the full support sums to ~1.
  virtual double integrate(double lo, double hi,
                           const ParentBins &parents) const = 0;

  // Clone for ownership purposes.
  virtual std::unique_ptr<CpdIntegrator> clone() const = 0;
};

// ----------------------------------------------------------------------------
// NormalCpd: X ~ N(mu(pa), sigma(pa))
//
// mu_fn and sigma_fn are computed from the ParentBins.  The closed-form
// integral uses the error function:
//   P(X in [lo, hi)) = 0.5 * (erf((hi-mu)/(sqrt(2) sigma))
//                            - erf((lo-mu)/(sqrt(2) sigma)))
// ----------------------------------------------------------------------------
class NormalCpd : public CpdIntegrator {
public:
  using Fn = std::function<double(const ParentBins &)>;
  NormalCpd(Fn mu_fn, Fn sigma_fn)
      : mu_fn_(std::move(mu_fn)), sigma_fn_(std::move(sigma_fn)) {}

  double integrate(double lo, double hi,
                   const ParentBins &parents) const override;

  std::unique_ptr<CpdIntegrator> clone() const override {
    return std::make_unique<NormalCpd>(mu_fn_, sigma_fn_);
  }

private:
  Fn mu_fn_;
  Fn sigma_fn_;
};

// ----------------------------------------------------------------------------
// UniformCpd: X ~ Uniform(a(pa), b(pa))
// ----------------------------------------------------------------------------
class UniformCpd : public CpdIntegrator {
public:
  using Fn = std::function<double(const ParentBins &)>;
  UniformCpd(Fn a_fn, Fn b_fn)
      : a_fn_(std::move(a_fn)), b_fn_(std::move(b_fn)) {}

  double integrate(double lo, double hi,
                   const ParentBins &parents) const override;

  std::unique_ptr<CpdIntegrator> clone() const override {
    return std::make_unique<UniformCpd>(a_fn_, b_fn_);
  }

private:
  Fn a_fn_;
  Fn b_fn_;
};

// ----------------------------------------------------------------------------
// ExponentialCpd: X ~ Exponential(rate(pa))
// Support [0, +inf).  P(X in [lo, hi)) = exp(-rate*lo) - exp(-rate*hi).
// Negative lo is treated as 0.
// ----------------------------------------------------------------------------
class ExponentialCpd : public CpdIntegrator {
public:
  using Fn = std::function<double(const ParentBins &)>;
  explicit ExponentialCpd(Fn rate_fn) : rate_fn_(std::move(rate_fn)) {}

  double integrate(double lo, double hi,
                   const ParentBins &parents) const override;

  std::unique_ptr<CpdIntegrator> clone() const override {
    return std::make_unique<ExponentialCpd>(rate_fn_);
  }

private:
  Fn rate_fn_;
};

// ----------------------------------------------------------------------------
// LogNormalCpd: log(X) ~ N(log_mu(pa), log_sigma(pa))
// Support (0, +inf).  Uses the erf identity on log-space.
// ----------------------------------------------------------------------------
class LogNormalCpd : public CpdIntegrator {
public:
  using Fn = std::function<double(const ParentBins &)>;
  LogNormalCpd(Fn log_mu_fn, Fn log_sigma_fn)
      : log_mu_fn_(std::move(log_mu_fn)),
        log_sigma_fn_(std::move(log_sigma_fn)) {}

  double integrate(double lo, double hi,
                   const ParentBins &parents) const override;

  std::unique_ptr<CpdIntegrator> clone() const override {
    return std::make_unique<LogNormalCpd>(log_mu_fn_, log_sigma_fn_);
  }

private:
  Fn log_mu_fn_;
  Fn log_sigma_fn_;
};

// ----------------------------------------------------------------------------
// DeterministicCpd: Y = g(parents).  Treated specially by the manager —
// the `integrate()` method here is a midpoint-collocation fallback.  The
// DiscretizationManager detects instances of this class via dynamic_cast
// and distributes mass across child bins using either:
//   (a) interval arithmetic (if monotone=true): evaluate g at the extreme
//       corners of the parent hyper-rect, spread mass by overlap length
//   (b) Monte-Carlo sampling (if monotone=false): average n_samples
//       evaluations of g across the parent hyper-rect
// Unlike the stochastic CPDs, row-normalisation is a no-op because the
// mass already comes out to 1.
// ----------------------------------------------------------------------------
class DeterministicCpd : public CpdIntegrator {
public:
  using Fn = std::function<double(const ParentBins &)>;
  // monotone = true  → exact distribution via interval arithmetic
  // monotone = false → Monte-Carlo sampling (n_samples evaluations)
  DeterministicCpd(Fn fn, bool monotone = false, std::size_t n_samples = 32)
      : fn_(std::move(fn)), monotone_(monotone), n_samples_(n_samples) {}

  // Fallback: midpoint collocation — returns 1 if g(parents) ∈ [lo, hi).
  double integrate(double lo, double hi,
                   const ParentBins &parents) const override {
    const double y = fn_(parents);
    return (y >= lo && y < hi) ? 1.0 : 0.0;
  }

  std::unique_ptr<CpdIntegrator> clone() const override {
    return std::make_unique<DeterministicCpd>(fn_, monotone_, n_samples_);
  }

  const Fn &fn() const { return fn_; }
  bool monotone() const { return monotone_; }
  std::size_t n_samples() const { return n_samples_; }

private:
  Fn fn_;
  bool monotone_;
  std::size_t n_samples_;
};

// ----------------------------------------------------------------------------
// UserFunctionCpd: the user supplies the integral directly.
// Signature: double fn(double lo, double hi, const ParentBins &)
// This is the escape hatch for densities outside the built-in set.
// ----------------------------------------------------------------------------
class UserFunctionCpd : public CpdIntegrator {
public:
  using Fn = std::function<double(double, double, const ParentBins &)>;
  explicit UserFunctionCpd(Fn fn) : fn_(std::move(fn)) {}

  double integrate(double lo, double hi,
                   const ParentBins &parents) const override {
    return fn_(lo, hi, parents);
  }

  std::unique_ptr<CpdIntegrator> clone() const override {
    return std::make_unique<UserFunctionCpd>(fn_);
  }

private:
  Fn fn_;
};

} // namespace bncore
