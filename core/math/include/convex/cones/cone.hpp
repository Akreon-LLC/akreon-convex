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

  /// @brief hessian type
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
   * y0 \in S^0(D)
   *
   * @return vector
   */
  constexpr Slack y0() const { return self().y0_impl(); }

  /**
   * @brief in the interior of k
   *
   * x \in K
   *
   * @param x \in K
   *
   * @return bool
   */
  constexpr bool in_cone(const Slack& x) const {
    return self().in_cone_impl(x);
  }

  /**
   * @brief in the interior of K^*
   *
   * y \in K^* ?
   *
   * @param y \in K^*
   *
   * @return bool
   */
  constexpr bool in_dual_cone(const Slack& y) const {
    return self().in_dual_cone_impl(y);
  }

  /**
   * @brief hessian
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return H
   */
  constexpr auto H(const Slack& x, const Slack& y) const {
    return self().H_impl(x, y);
  }

  /**
   * @brief fraction to boundary dk
   *
   * \sigma_x(p) = \frac{1}{\sup{\alpha | x - \alpha p \in k }}
   * \alpha_max = \frac{1}{\sigma_x(p)}
   *
   * we want the largest alpha that stays within k for our next step
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
   * @brief fraction to boundary dk^*
   *
   * \sigma_x(p) = \frac{1}{\sup{\alpha | y - \alpha p \in k^* }}
   * \alpha_max = \frac{1}{\sigma_x(p)}
   *
   * we want the largest alpha that stays within k for our next step
   * y_{k+1} = y_{k} - \alpha p
   *
   * @param y \in K^*
   * @param p \in E^*
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_dual(const Slack& y,
                             const linalg::Vector<T, M>& p) const {
    return self().alpha_max_dual_impl(y, p);
  }

  /**
   * @brief barrier parameter
   *
   * For jordan algebra it is associated with the dimension of the cone
   *
   * @return nu
   */
  constexpr T nu() const { return self().nu_impl(); }

  /**
   * @brief gradient
   *
   * F'(x)
   *
   * @param x \in K
   *
   * @return F'(x)
   */
  constexpr auto grad(const Slack& x) const { return self().grad_impl(x); }

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
   * @brief w (self-scaling point)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return w
   */
  constexpr auto w(const Slack& x, const Slack& y) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().w_impl(x, y);
  }

  /**
   * @brief circ (jordan map/product operator)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x \circ y
   */
  constexpr auto circ(const Slack& x, const Slack& y) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().circ_impl(x, y);
  }

  /**
   * @brief trace
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return tr(x \circ y)
   */
  constexpr auto trace(const Slack& x, const Slack& y) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().trace_impl(x, y);
  }

  /**
   * @brief inv (jordan inverse element)
   *
   * @param x \in K or y \in K^*
   *
   * @return x^{-1}
   */
  constexpr auto inv(const Slack& x) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().inv_impl(x);
  }

  /**
   * @brief \gamma_F (functional proximity neighborhood metric)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return \gamma_F
   */
  constexpr auto gamma_F(const Slack& x, const Slack& y) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().gamma_F_impl(x, y);
  }

  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }
};

}  // namespace conic
