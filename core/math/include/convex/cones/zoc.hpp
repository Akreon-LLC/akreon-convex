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

/** @brief zoc cone class
 *
 * K_zoc = { x | x == 0 }
 */
template <linalg::VectorLike Slack>
struct ZeroOrderCone
    : public Cone<Slack, ZeroOrderCone<Slack>, ConeType::SYMMETRIC> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, ZeroOrderCone<Slack>, ConeType::SYMMETRIC>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::HessianMat;
  using typename ConeBase::T;

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const { return Slack{}; }

  /**
   * @brief default dual point
   *
   * @return vector
   */
  constexpr Slack y0_impl() const { return Slack{}; }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& /*x*/) const {
    /// (dakre) always empty \equiv null set!
    return false;
  }

  /**
   * @brief in interior of K^*
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_dual_cone_impl(const Slack& /*y*/) const { return false; }

  /**
   * @brief fraction to boundary dK
   *
   * @param x \in K
   * @param p \in E
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_impl(const Slack& /*x*/,
                             const linalg::Vector<T, M>& /*p*/) const {
    return T(1);
  }

  /**
   * @brief fraction to boundary dK^*
   *
   * @param y \in K^*
   * @param p \in E^*
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_dual_impl(const Slack& /*y*/,
                                  const linalg::Vector<T, M>& /*p*/) const {
    return T(1);
  }

  /**
   * @brief hessian
   *
   * Zero cone carries no barrier geometry, so return the zero operator.
   *
   * @return H
   */
  constexpr auto H_impl(const Slack& /*x*/, const Slack& /*y*/) const {
    return linalg::Hessian<HessianMat>(HessianMat{});
  }

  /**
   * @brief barrier parameter
   *
   * @return nu
   */
  constexpr T nu_impl() const { return T(0); }

  /**
   * @brief gradient
   *
   * @return zero vector
   */
  constexpr auto grad_impl(const Slack& /*x*/) const { return Slack{}; }

  /**
   * @brief jordan identity
   *
   * @return zero vector
   */
  constexpr auto e_impl() const { return Slack{}; }

  /**
   * @brief self-scaled point
   *
   * @return zero vector
   */
  constexpr auto w_impl(const Slack& /*x*/, const Slack& /*y*/) const {
    return Slack{};
  }

  /**
   * @brief jordan product
   *
   * @return zero vector
   */
  constexpr auto circ_impl(const Slack& /*x*/, const Slack& /*y*/) const {
    return Slack{};
  }

  /**
   * @brief trace
   *
   * @return zero scalar
   */
  constexpr auto trace_impl(const Slack& /*x*/, const Slack& /*y*/) const {
    return T(0);
  }

  /**
   * @brief jordan inverse
   *
   * @return zero vector
   */
  constexpr auto inv_impl(const Slack& /*x*/) const { return Slack{}; }

  /**
   * @brief functional proximity metric
   *
   * @return zero scalar
   */
  constexpr auto gamma_F_impl(const Slack& /*x*/, const Slack& /*y*/) const {
    return T(0);
  }
};

}  // namespace conic
