#include "bncore/discretization/cpd_integrator.hpp"
#include <algorithm>
#include <cmath>

namespace bncore {

// Standard-normal CDF using erf: Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
static inline double phi_cdf(double z) {
  return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// ----------------------------------------------------------------------------
// NormalCpd
// ----------------------------------------------------------------------------
double NormalCpd::integrate(double lo, double hi,
                            const ParentBins &parents) const {
  const double mu = mu_fn_(parents);
  const double sigma = sigma_fn_(parents);
  if (sigma <= 0.0 || !std::isfinite(sigma) || !std::isfinite(mu)) return 0.0;
  if (hi <= lo) return 0.0;
  return phi_cdf((hi - mu) / sigma) - phi_cdf((lo - mu) / sigma);
}

// ----------------------------------------------------------------------------
// UniformCpd — P(X in [lo,hi)) = overlap with [a,b) / (b-a)
// ----------------------------------------------------------------------------
double UniformCpd::integrate(double lo, double hi,
                             const ParentBins &parents) const {
  const double a = a_fn_(parents);
  const double b = b_fn_(parents);
  if (b <= a || hi <= lo) return 0.0;
  const double ov_lo = std::max(lo, a);
  const double ov_hi = std::min(hi, b);
  if (ov_hi <= ov_lo) return 0.0;
  return (ov_hi - ov_lo) / (b - a);
}

// ----------------------------------------------------------------------------
// ExponentialCpd — P(X in [lo,hi)) = F(hi) - F(lo), F(x) = 1 - exp(-rate x)
// Negative lo clamped to 0 (support is [0, +inf)).
// ----------------------------------------------------------------------------
double ExponentialCpd::integrate(double lo, double hi,
                                 const ParentBins &parents) const {
  const double rate = rate_fn_(parents);
  if (rate <= 0.0 || !std::isfinite(rate) || hi <= lo) return 0.0;
  const double a = std::max(lo, 0.0);
  const double b = std::max(hi, 0.0);
  if (b <= a) return 0.0;
  return std::exp(-rate * a) - std::exp(-rate * b);
}

// ----------------------------------------------------------------------------
// LogNormalCpd — P(X in [lo,hi)) via erf on log-space.
// Support (0, +inf).  If lo <= 0, use log(max(lo, tiny)) and include
// all mass below the first positive bin edge.
// ----------------------------------------------------------------------------
double LogNormalCpd::integrate(double lo, double hi,
                               const ParentBins &parents) const {
  const double mu = log_mu_fn_(parents);
  const double sigma = log_sigma_fn_(parents);
  if (sigma <= 0.0 || !std::isfinite(sigma) || !std::isfinite(mu)) return 0.0;
  if (hi <= lo) return 0.0;

  // CDF on log-space: Phi((log(x) - mu) / sigma).
  auto cdf = [&](double x) -> double {
    if (x <= 0.0) return 0.0;
    return phi_cdf((std::log(x) - mu) / sigma);
  };
  return cdf(hi) - cdf(lo);
}

} // namespace bncore
