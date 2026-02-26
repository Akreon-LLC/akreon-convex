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
    HessianMat H{};
    for (size_t i = 0; i < M; ++i) {
      const auto val = linalg::constexpr_abs(x(i)) < eps ? eps : x(i);
      H(i)(i) = s(i) / val;
    }
    return linalg::Hessian(H);
  }
  /**
   * @brief hessian inverse
   *
   *    F''(w)^{-1} -> pullback operator that maps vectors in the gradient space
   * back to primal / tangent dirs... acts as metric induced lifting
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return H^{-1}
   */
  constexpr auto H_inv_impl(const Slack& x, const Slack& s) const {
    HessianMat H_inv{};
    for (size_t i = 0; i < M; ++i) {
      const auto val = linalg::constexpr_abs(s(i)) < eps ? eps : s(i);
      H_inv(i)(i) = x(i) / val;
    }
    return linalg::Hessian<HessianMat, linalg::DifferentialType::PULLBACK>(
        H_inv);
  }

  /**
   * @brief barrier parameter
   *
   * @return nu
   */
  constexpr T nu_impl() const { return static_cast<T>(M); }

  /**
   * @brief circ (jordan map/product)
   *
   * x \circ s = x_i s_i (componentwise mul)
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return x \circ s (vector)
   */
  constexpr auto circ_impl(const Slack& x, const Slack& s) const {
    Slack res{};
    for (size_t i = 0; i < M; ++i) res(i) = x(i) * s(i);
    return res;
  }

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
};

}  // namespace conic
