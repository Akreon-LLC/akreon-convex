/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include "cone.hpp"
#include "linalg/operator/quadratic_rep.hpp"

namespace conic {

/**
 * @brief Circ - Jordan Map/Product for R^n_+
 *
 * x circ y = x_i y_i
 */
template <linalg::VectorContainer Vec>
struct RplusCirc {
  /// @brief types and size extraction
  using T = linalg::vector_element_t<Vec>;
  static constexpr std::size_t N = linalg::vector_dim_v<Vec>;

  /**
   * @brief circ (jordan map/product)
   *
   * x circ y = x_i y_i (componentwise mul)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return x circ y (vector)
   */
  constexpr auto operator()(const Vec& x, const Vec& y) const {
    Vec res{};
    for (std::size_t i = 0; i < N; ++i) res(i) = x(i) * y(i);
    return res;
  }
};

/** @brief nonneg cone class
 *
 * R^n_+ = set of all nonneg R
 */
template <linalg::VectorContainer Slack>
struct NonnegOrthantCone
    : public Cone<Slack, NonnegOrthantCone<Slack>, ConeType::SYMMETRIC> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, NonnegOrthantCone<Slack>, ConeType::SYMMETRIC>;

  /// @brief helpers to get T and M
  using ConeBase::eps;
  using ConeBase::M;
  using typename ConeBase::HessianMat;
  using typename ConeBase::T;

  /**
   * @brief default interior point
   *
   * @return x0
   */
  constexpr Slack x0_impl() const {
    Slack s{};
    for (std::size_t i = 0; i < M; ++i) s(i) = T(1);
    return s;
  }

  /**
   * @brief default dual point
   *
   * @return y0
   */
  constexpr Slack y0_impl() const { return x0_impl(); }

  /**
   * @brief in interior of K
   *
   * @param x in K
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    for (std::size_t i = 0; i < M; ++i)
      if (!(x(i) > T(0))) return false;
    return true;
  }

  /**
   * @brief in interior of K^*
   *
   * @param y in K^*
   *
   * @return bool
   */
  constexpr bool in_dual_cone_impl(const Slack& y) const {
    return in_cone_impl(y);
  }

  /**
   * @brief fraction to boundary dK
   *
   * sigma_x = frac{1}{sup{alpha | x + alpha p in K}}
   * sup{alpha | x + alpha p in K} -> -inf(x + alpha p)
   * x + alpha p = 0 -> alpha = -x / p
   * alpha = min. -x_i / p_i
   *
   * (similar for dual)
   *
   * @param x in K or K^*
   * @param p in E or E^* (delta dir)
   *
   * @return alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    constexpr T eta{0.99};
    T a{1};
    for (std::size_t i = 0; i < M; ++i) {
      if (p(i) < T(0)) {
        const T cand = -x(i) / p(i);
        a = (cand < a) ? cand : a;
      }
    }
    return std::min(T(1), eta * a);
  }

  /**
   * @brief fracto to dK^*
   *
   * @param y in K
   * @param p in E^*
   *
   * @return alpha_max (dual)
   */
  constexpr T alpha_max_dual_impl(const Slack& y,
                                  const linalg::Vector<T, M>& p) const {
    return alpha_max_impl(y, p);
  }

  /**
   * @brief hessian
   *
   * F''(w) -> pushforward operator that maps primal / tangent dir to change in
   * gradient at scaling point w... acts as forward sensitivity & curvature
   *
   * w_i = sqrt(x_i/y_i)
   * F(w) = -log(sum(w_i))
   * F'(w) = -(1/w)^{T}
   * F''(w) = diag(1/w^2)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return (H, event)
   */
  constexpr auto H_impl(const Slack& x, const Slack& y) const {
    const auto [w, event] = w_impl(x, y);
    if (linalg::is_fatal_event(event))
      return std::tuple(
          linalg::QuadraticRep<HessianMat, RplusCirc<Slack>>(e_impl()), event);
    Slack w_inv{};
    bool safeguarded = false;
    for (std::size_t i = 0; i < M; ++i) {
      const T wi_safe = std::max(w(i), eps);
      safeguarded = safeguarded || (wi_safe != w(i));
      w_inv(i) = T(1) / wi_safe;
    }
    const auto out_event =
        linalg::is_fatal_event(event) ? event
        : (event != linalg::NumericalEvent::NONE || safeguarded)
            ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
            : linalg::NumericalEvent::NONE;
    return std::tuple(linalg::QuadraticRep<HessianMat, RplusCirc<Slack>>(w_inv),
                      out_event);
  }

  /**
   * @brief grad
   *
   * F'(x) = -(1/x)^{T}
   *
   * @param x in K or y in K^*
   *
   * @return (F'(x), event)
   */
  constexpr auto grad_impl(const Slack& x) const {
    Slack dx{};
    bool safeguarded = false;
    for (std::size_t i = 0; i < M; ++i) {
      const T xi_safe = std::max(x(i), eps);
      safeguarded = safeguarded || (xi_safe != x(i));
      dx(i) = T(-1) / xi_safe;
    }
    return std::tuple(dx, safeguarded
                              ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
                              : linalg::NumericalEvent::NONE);
  }

  /**
   * @brief barrier parameter
   *
   * @return nu
   */
  constexpr T nu_impl() const { return static_cast<T>(M); }

  /**
   * @brief e (jordan identity)
   *
   * e = (1,1,...1)
   *
   * @return e (vector)
   */
  constexpr auto e_impl() const {
    linalg::vector_traits<Slack> vec{};
    return vec.fill(T(1));
  }

  /**
   * @brief w (self-scaled point)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return (w, event)
   */
  constexpr auto w_impl(const Slack& x, const Slack& y) const {
    Slack w{};
    bool safeguarded{false};
    for (std::size_t i = 0; i < M; ++i) {
      const T x_safe = std::max(x(i), eps);
      const T y_safe = std::max(y(i), eps);
      safeguarded = safeguarded || (x_safe != x(i)) || (y_safe != y(i));
      w(i) = linalg::constexpr_sqrt_safe(x_safe / y_safe);
    }
    return std::tuple(w, safeguarded
                             ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
                             : linalg::NumericalEvent::NONE);
  }

  /**
   * @brief circ
   *
   * x circ y
   *
   * @param x in K
   * @param y in K^*
   *
   * @return x circ y
   */
  constexpr auto circ_impl(const Slack& x, const Slack& y) const {
    return RplusCirc<Slack>()(x, y);
  }

  /**
   * @brief Componentwise form of the NT-scaled affine correction for R^n_+.
   *
   * Here x circ y is componentwise multiplication. Complementarity gives
   * y_i*dx_i + x_i*dy_i = mu - x_i*y_i - px_i*py_i.
   * Dividing by x_i gives
   * dy_i = -(y_i/x_i)*dx_i - [y_i - mu/x_i + px_i*py_i/x_i].
   * Thus the reduced dual residual needs px_i*py_i/x_i, not the raw product.
   * The existing interior safeguard bounds the denominator away from zero.
   */
  template <typename Scaling>
  constexpr auto scaled_affine_correction_impl(const Slack& x, const Scaling&,
                                               const Slack& px,
                                               const Slack& py) const {
    Slack correction{};
    bool safeguarded = false;
    for (std::size_t i = 0; i < M; ++i) {
      const T xi_safe = std::max(x(i), eps);
      safeguarded = safeguarded || (xi_safe != x(i));
      correction(i) = px(i) * py(i) * (T(1) / xi_safe);
    }
    return std::tuple(correction,
                      safeguarded ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
                                  : linalg::NumericalEvent::NONE);
  }

  /**
   * @brief trace
   *
   * trace(x circ y) = sum(x_i y_i)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return x circ y
   */
  constexpr auto trace_impl(const Slack& x, const Slack& y) const {
    const auto x_circ_y = circ_impl(x, y);
    T res{};
    for (std::size_t i = 0; i < M; ++i) res += x_circ_y(i);
    return res;
  }

  /**
   * @brief jordan inverse
   *
   * @param x in K or y in K^*
   *
   * @return (inv(x), event)
   */
  constexpr auto inv_impl(const Slack& x) const {
    Slack res{};
    bool safeguarded = false;
    for (std::size_t i = 0; i < M; ++i) {
      const T xi_safe = std::max(x(i), eps);
      safeguarded = safeguarded || (xi_safe != x(i));
      res(i) = T(1) / xi_safe;
    }
    return std::tuple(res, safeguarded
                               ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
                               : linalg::NumericalEvent::NONE);
  }

  /**
   * @brief gamma_F
   *
   * @param x in K
   * @param y in K^*
   *
   * @return gamma_F
   */
  constexpr auto gamma_F_impl(const Slack& x, const Slack& y) const {
    T res{};
    bool safeguarded = false;
    for (std::size_t i = 0; i < M; ++i) {
      const auto xy_safe = std::max(x(i) * y(i), eps);
      safeguarded = safeguarded || (xy_safe != x(i) * y(i));
      res += linalg::constexpr_ln_safe(xy_safe, eps);
    }
    const auto xTy_safe =
        std::max(linalg::inner_product(x, y), eps) / nu_impl();
    safeguarded =
        safeguarded || (xTy_safe * nu_impl() != linalg::inner_product(x, y));
    res = nu_impl() * linalg::constexpr_ln_safe(xTy_safe, eps) - res;
    return std::tuple(res, safeguarded
                               ? linalg::NumericalEvent::NUMERICAL_SAFEGUARD
                               : linalg::NumericalEvent::NONE);
  }
};

} // namespace conic
