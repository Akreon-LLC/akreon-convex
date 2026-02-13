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

namespace conic {

/** @brief nonneg cone class
 *
 * R^n_+ = set of all nonneg R
 */
template <linalg::VectorLike Slack>
struct NonnegOrthantCone : public Cone<Slack, NonnegOrthantCone<Slack>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, NonnegOrthantCone<Slack>>;

  /// @brief helpers to get T and M
  using ConeBase::eps;
  using ConeBase::M;
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
   * @return s0
   */
  constexpr Slack s0_impl() const { return x0_impl(); }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    for (size_t i = 0; i < M; ++i)
      if (x(i) <= T(0)) return false;
    return true;
  }

  /**
   * @brief fraction to boundary dK
   *
   * sigma_x = \frac{1}{sup{alpha | x - alpha p \in K}}
   * sup{alpha | x - alpha p \in K} = 1 / sigma_x(p)
   * alpha = min. x_i / p_i
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
    T mn = std::numeric_limits<T>::max();
    for (size_t i = 0; i < M; ++i) {
      if (linalg::constexpr_abs(p(i)) < eps) continue;
      mn = x(i) / p(i) < mn ? x(i) / p(i) : mn;
    }
    return mn;
  }

  /**
   * @brief hessian
   *
   * F''(w) -> pushforward operator that maps primal / tangent dir to change in
   * gradient at scaling point w... acts as forward sensitivity & curvature
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return H
   */
  constexpr auto H_impl(const Slack& x, const Slack& s) const {
    linalg::Matrix<T, M, M> H{};
    for (size_t i = 0; i < M; ++i) {
      const auto val = linalg::constexpr_abs(x(i)) < eps ? eps : x(i);
      H(i)(i) = s(i) / val;
    }
    return linalg::Hessian(H);
  }
  /**
   * @brief hessian inverse
   *
   * F''(w)^{-1} -> pullback operator that maps vectors in the gradient space
   * back to primal / tangent dirs... acts as metric induced lifting
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return H
   */
  constexpr auto H_inv_impl(const Slack& x, const Slack& s) const {
    linalg::Matrix<T, M, M> H_inv{};
    for (size_t i = 0; i < M; ++i) {
      const auto val = linalg::constexpr_abs(s(i)) < eps ? eps : s(i);
      H_inv(i)(i) = x(i) / val;
    }
    return linalg::Hessian<decltype(H_inv), linalg::DifferentialType::PULLBACK>(
        H_inv);
  }

  /**
   * @brief gradient
   *
   * \nabla F(x) -> maps point in primal space to cotangent / dual space of
   * vectors (i.e., covectors)
   *
   * @param x \in K
   *
   * @return grad
   */
  constexpr auto grad_impl(const Slack& x) const {
    Slack grad{};
    for (size_t i = 0; i < M; ++i) {
      const auto val = linalg::constexpr_abs(x(i)) < eps ? eps : x(i);
      grad(i) = T(-1) / val;
    }
    return grad;
  }

  /**
   * @brief functional proximity
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return functional proximity measure
   */
  constexpr auto functional_proximity_impl(const Slack& x,
                                           const Slack& s) const {
    T gamma_F{0};
    for (size_t i = 0; i < M; ++i) {
      const auto val = x(i) * s(i) < eps ? eps : x(i) * s(i);
      gamma_F -= linalg::constexpr_ln(val).value();
    }

    const auto xs = linalg::dot(x, s);
    const auto val = xs < eps ? eps : xs;
    gamma_F += M * linalg::constexpr_ln(val).value();
    return gamma_F;
  }

  /**
   * @brief duality gap (normalized)
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return \mu(x,s)
   */
  constexpr T mu_impl(const Slack& x, const Slack& s) const {
    return linalg::dot(x, s) / static_cast<T>(M);
  }
};

}  // namespace conic
