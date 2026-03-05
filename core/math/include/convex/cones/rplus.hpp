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
 * x \circ y = x_i y_i
 */
template <linalg::VectorLike Vec>
struct RplusCirc {
  /// @brief types and size extraction
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;

  /**
   * @brief circ (jordan map/product)
   *
   * x \circ y = x_i y_i (componentwise mul)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x \circ y (vector)
   */
  constexpr auto operator()(const Vec& x, const Vec& y) const {
    Vec res{};
    for (size_t i = 0; i < N; ++i) res(i) = x(i) * y(i);
    return res;
  }
};

/** @brief nonneg cone class
 *
 * R^n_+ = set of all nonneg R
 */
template <linalg::VectorLike Slack>
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
    for (size_t i = 0; i < M; ++i) s(i) = T(1);
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
   * @param x \in K
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    for (size_t i = 0; i < M; ++i) {
      if (!(x(i) > T(0))) return false;
    }
    return true;
  }

  /**
   * @brief in interior of K^*
   *
   * @param y \in K^*
   *
   * @return bool
   */
  constexpr bool in_dual_cone_impl(const Slack& y) const {
    return in_cone_impl(y);
  }

  /**
   * @brief fraction to boundary dK
   *
   * sigma_x = \frac{1}{sup{alpha | x + alpha p \in K}}
   * sup{alpha | x + alpha p \in K} -> -inf(x + alpha p)
   * x + alpha p = 0 -> alpha = -x / p
   * alpha = min. -x_i / p_i
   *
   * (similar for dual)
   *
   * @param x \in K or K^*
   * @param p \in E or E^* (delta dir)
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    constexpr T eta{0.99};
    T a{1};
    for (size_t i = 0; i < M; ++i) {
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
   * @param y \in K
   * @param p \in E^*
   *
   * @return \alpha_max (dual)
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
   * @param x \in K
   * @param y \in K^*
   *
   * @return H
   */
  constexpr auto H_impl(const Slack& x, const Slack& y) const {
    const auto w = w_impl(x, y);
    Slack w_inv{};
    for (size_t i = 0; i < M; ++i) {
      const T wi_safe = std::max(w(i), eps);
      w_inv(i) = T(1) / wi_safe;
    }
    return linalg::QuadraticRep<HessianMat, RplusCirc<Slack>>(w_inv);
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
    linalg::vec_traits<Slack> vec{};
    return vec.fill(T(1));
  }

  /**
   * @brief w (self-scaled point)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return w
   */
  constexpr auto w_impl(const Slack& x, const Slack& y) const {
    Slack w{};
    for (size_t i = 0; i < M; ++i) {
      const T x_safe = std::max(x(i), eps);
      const T y_safe = std::max(y(i), eps);
      w(i) = linalg::constexpr_sqrt_safe(x_safe / y_safe);
    }
    return w;
  }

  /**
   * @brief circ
   *
   * x \circ y
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x circ y
   */
  constexpr auto circ_impl(const Slack& x, const Slack& y) const {
    return RplusCirc<Slack>()(x, y);
  }

  /**
   * @brief trace
   *
   * trace(x \circ y) = sum(x_i y_i)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x circ y
   */
  constexpr auto trace_impl(const Slack& x, const Slack& y) const {
    const auto x_circ_y = circ_impl(x, y);
    T res{};
    for (size_t i = 0; i < M; ++i) res += x_circ_y(i);
    return res;
  }

  /**
   * @brief gamma_F
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return gamma_F
   */
  constexpr auto gamma_F_impl(const Slack& x, const Slack& y) const {
    T res{};
    for (size_t i = 0; i < M; ++i) {
      const auto xy_safe = std::max(x(i) * y(i), eps);
      res += linalg::constexpr_ln_safe(xy_safe, eps);
    }
    const auto xTy_safe = std::max(linalg::dot(x, y), eps) / nu_impl();
    res = nu_impl() * linalg::constexpr_ln_safe(xTy_safe, eps) - res;
    return res;
  }
};

}  // namespace conic
