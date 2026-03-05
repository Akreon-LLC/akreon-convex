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

#include "soc.hpp"

namespace conic {

/**
 * @brief rsoc cone class
 *
 * K_rsoc = { (u, v, w) | ||u||_2^2 <= 2 v w, v >= 0, w >= 0 }
 * K_soc = { (x, t) | ||x|| <= t, t >= 0 }
 *
 * RSOC -> SOC map
 * SOC: x^T x <= t^2... 0 <= t^2 - x^T x
 * RSOC: u^T u <= 2 v w... 0 <= 2 v w - u^T u
 *
 * id: (v + w)^2 - (v - w)^2 = 4 v w
 * linear change of vars -> (x, x0, t) \in R^{n+1} x R
 * t = v + w
 * x0 = v - w
 * x = sqrt(2)u
 *
 * SOC: (v - w)^2 + 2 u^T u <= (v + w)^2
 * v^2 - 2vw + w^2 + 2 u^T u <= v^2 + 2vw + w^2
 * 2u^T u <= 4vw
 * u^T u <= 2 vw (eq RSOC)
 *
 * A = [ sqrt(2)I 0  0 ] [ u; v; w ] = (x=sqrt(2)u, x0=v - w, t=v+w)
 *     [ 0        1 -1 ]
 *     [ 0        1  1 ]
 */
template <linalg::VectorLike Slack>
struct RotatedSecondOrderCone
    : public Cone<Slack, RotatedSecondOrderCone<Slack>,
                  ConeType::SYMMETRIC_WRAPPER> {
  /// @brief bring into scope alias from cone base
  using ConeBase =
      Cone<Slack, RotatedSecondOrderCone<Slack>, ConeType::SYMMETRIC_WRAPPER>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::HessianMat;
  using typename ConeBase::T;
  using Mat = linalg::Matrix<T, M, M>;

  static_assert(M >= 3, "RSOC requires N + 2");

  constexpr explicit RotatedSecondOrderCone() {
    const auto I = linalg::make_identity<T, M - 2>();
    Mat A{};
    A = linalg::set_block<0, M - 2, 0, M - 2>(
        A, linalg::constexpr_sqrt_safe(T(2)) * I);
    A(M - 2)(M - 2) = T(1);
    A(M - 2)(M - 1) = T(-1);
    A(M - 1)(M - 2) = T(1);
    A(M - 1)(M - 1) = T(1);

    map = affine::AffineMap<Mat>(A);

    Mat A_inv{};
    A_inv = linalg::set_block<0, M - 2, 0, M - 2>(
        A_inv, T(1) / linalg::constexpr_sqrt_safe(T(2)) * I);
    A_inv(M - 2)(M - 2) = T(0.5);
    A_inv(M - 2)(M - 1) = T(0.5);
    A_inv(M - 1)(M - 2) = T(-0.5);
    A_inv(M - 1)(M - 1) = T(0.5);

    inv_map = affine::AffineMap<Mat>(A_inv);

    const auto adj_inv_A = linalg::transpose(inv_map.linear_part());
    dual_map = affine::AffineMap<Mat>(adj_inv_A);
  }

  /**
   * @brief default interior point K
   *
   * @return x \in K
   */
  constexpr Slack x0_impl() const {
    Slack x{};
    x(M - 2) = T(1);
    x(M - 1) = T(2);
    return x;
  }

  /**
   * @brief default interior point K^*
   *
   * @return y \in K^*
   */
  constexpr Slack y0_impl() const { return x0_impl(); }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    const auto x_soc = map(x);
    return soc.in_cone(x_soc);
  }

  /**
   * @brief in interior of K^*
   *
   * @param y \in K^*
   *
   * @return bool
   */
  constexpr bool in_dual_cone_impl(const Slack& y) const {
    const auto y_soc = dual_map(y);
    return soc.in_dual_cone(y_soc);
  }

  /**
   * @brief fraction to boundary dK (max step)
   *
   * let x = (u, v, w)
   *
   * @param x \in K
   * @param p \in E (delta dir)
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    const auto x_soc = map(x);
    const auto p_soc = map(p);
    return soc.alpha_max(x_soc, p_soc);
  }

  /**
   * @brief fraction to boundary dK^* (max step)
   *
   * let x = (u, v, w)
   *
   * @param y \in K^*
   * @param p \in E^* (delta dir)
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_dual_impl(const Slack& y,
                                  const linalg::Vector<T, M>& p) const {
    const auto y_soc = dual_map(y);
    const auto p_soc = dual_map(p);
    return soc.alpha_max_dual(y_soc, p_soc);
  }

  /**
   * @brief hessian
   *
   * Pullback congruence map into RSOC coords
   * H_rsoc = A^T H_soc A
   * - A pushes RSOC primal dir into SOC primal dir
   * - H_soc acts in SOC
   * - A^T pulls the resulting SOC covector back into RSOC
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return H
   */
  constexpr auto H_impl(const Slack& x, const Slack& y) const {
    const auto x_soc = map(x);
    const auto y_soc = dual_map(y);
    const auto A = map.linear_part();
    const auto A_t = linalg::transpose(A);
    const auto H_soc = soc.H(x_soc, y_soc);

    HessianMat H_rsoc{};
    for (size_t j = 0; j < M; ++j) {
      const auto col = A_t * H_soc.apply(linalg::get_col(A, j));
      for (size_t i = 0; i < M; ++i) H_rsoc(i)(j) = col(i);
    }

    return linalg::Hessian<HessianMat>(H_rsoc);
  }

  /**
   * @brief barrier param
   *
   * @return nu
   */
  constexpr T nu_impl() const { return T(2); }

  /**
   * @brief e (jordan identity) in RSOC coords
   *
   * @return e
   */
  constexpr auto e_impl() const { return inv_map(soc.e()); }

  /**
   * @brief w (w-scaling point)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return w
   */
  constexpr auto w_impl(const Slack& x, const Slack& y) const {
    const auto x_soc = map(x);
    const auto y_soc = dual_map(y);
    return inv_map(soc.w(x_soc, y_soc));
  }

  /**
   * @brief circ
   *
   * x \circ y
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x \circ y
   */
  constexpr auto circ_impl(const Slack& x, const Slack& y) const {
    const auto x_soc = map(x);
    const auto y_soc = dual_map(y);
    return inv_map(soc.circ(x_soc, y_soc));
  }

  /**
   * @brief trace
   *
   * trace(x \circ y) = last elem
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return (x circ y)(-1)
   */
  constexpr auto trace_impl(const Slack& x, const Slack& y) const {
    const auto x_soc = map(x);
    const auto y_soc = dual_map(y);
    return soc.trace(x_soc, y_soc);
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
    const auto x_soc = map(x);
    const auto y_soc = dual_map(y);
    return soc.gamma_F(x_soc, y_soc);
  }

  /** @brief member var */
  SecondOrderCone<Slack> soc;       ///< native SOC type
  affine::AffineMap<Mat> map;       ///< RSOC -> SOC map (primal)
  affine::AffineMap<Mat> inv_map;   ///< SOC -> RSOC map (primal inverse)
  affine::AffineMap<Mat> dual_map;  ///< RSOC -> SOC map (dual)
};

}  // namespace conic
