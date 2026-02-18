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

#include <climits>
#include <optional>

#include "convex/cone_traits.hpp"
#include "linalg/operator/hessian.hpp"

namespace conic {

/// @brief cone base class
template <linalg::VectorLike Slack, typename Derived, ConeType CType>
struct Cone {
  /// @brief alias for mat type
  using T = linalg::vec_type_t<Slack>;

  /// @brief helpers to get M slack size
  static constexpr size_t M = linalg::vec_size_v<Slack>;

  /// @brief eps constant
  static constexpr T eps = 1e-8;

  /// @brief HessianMat
  using HessianMat = linalg::Matrix<T, M, M>;

  /**
   * @brief default interior point (initial slack in cone space)
   *
   * x0 \in S^0(P)
   *
   * @return vector
   */
  constexpr Slack x0() const { return self().x0_impl(); }

  /**
   * @brief default interior point (initial slack in dual cone)
   *
   * s0 \in S^0(D)
   *
   * @return vector
   */
  constexpr Slack s0() const { return self().s0_impl(); }

  /**
   * @brief in the interior of k
   *
   * x \in K ? or (s,y) \in K^* ?
   *
   * @param x \in K
   *
   * @return bool
   */
  constexpr bool in_cone(const Slack& x) const {
    return self().in_cone_impl(x);
  }

  /**
   * @brief hessian
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return H
   */
  constexpr auto H(const Slack& x, const Slack& s) const {
    return self().H_impl(x, s);
  }

  /**
   * @brief hessian inverse
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return H
   */
  constexpr auto H_inv(const Slack& x, const Slack& s) const {
    return self().H_inv_impl(x, s);
  }

  /**
   * @brief fraction to boundary dK
   *
   * \sigma_x(p) = \frac{1}{\sup{\alpha | x - \alpha p \in K }}
   * \alpha_max = \frac{1}{\sigma_x(p)}
   *
   * we want the largest alpha that stays within K for our next step
   * x_{k+1} = x_{k} - \alpha p
   *
   * @param x \in K
   * @param p \in E
   *
   * @return \alpha_max
   */
  constexpr T alpha_max(const Slack& x, const linalg::Vector<T, M>& p) const {
    return self().alpha_max_impl(x, p);
  }

  /**
   * @brief duality gap (normalized)
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return \mu(x,s)
   */
  constexpr T mu(const Slack& x, const Slack& s) const {
    return self().mu_impl(x, s);
  }

  /**
   * @brief circ (jordan product / map)
   *
   * @param x \in K
   * @param s \in K^*
   *
   * @return x \circ s
   */
  constexpr auto circ(const Slack& x, const Slack& s) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().circ_impl(x, s);
  }

  /**
   * @brief e (jordan identity)
   *
   * x \circ e = x
   *
   * @return e
   */
  constexpr auto e() const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().e_impl();
  }

  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }
};

}  // namespace conic
