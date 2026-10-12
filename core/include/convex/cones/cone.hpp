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
#include "linalg/operator/ldl.hpp"

namespace conic {

/// @brief cone base class
template <linalg::VectorContainer Slack, typename Derived, ConeType CType>
struct Cone {
  /// @brief alias for mat type
  using T = linalg::vector_element_t<Slack>;

  /// @brief helpers to get M slack size
  static constexpr std::size_t M = linalg::vector_dim_v<Slack>;

  /// @brief eps constant
  static constexpr T eps = 1e-8;

  /// @brief hessian type
  using HessianMat = linalg::Matrix<T, M, M>;

  /**
   * @brief default interior point (initial slack in cone space)
   *
   * x0 in S^0(P)
   *
   * @return vector
   */
  constexpr Slack x0() const { return self().x0_impl(); }

  /**
   * @brief default interior point (initial slack in dual cone)
   *
   * y0 in S^0(D)
   *
   * @return vector
   */
  constexpr Slack y0() const { return self().y0_impl(); }

  /**
   * @brief in the interior of k
   *
   * x in K
   *
   * @param x in K
   *
   * @return bool
   */
  constexpr bool in_cone(const Slack& x) const {
    return self().in_cone_impl(x);
  }

  /**
   * @brief in the interior of K^*
   *
   * y in K^* ?
   *
   * @param y in K^*
   *
   * @return bool
   */
  constexpr bool in_dual_cone(const Slack& y) const {
    return self().in_dual_cone_impl(y);
  }

  /**
   * @brief hessian
   *
   * @param x in K
   * @param y in K^*
   *
   * @return (H, event)
   */
  constexpr auto H(const Slack& x, const Slack& y) const {
    return self().H_impl(x, y);
  }

  /**
   * @brief fraction to boundary dk
   *
   * sigma_x(p) = frac{1}{sup{alpha | x - alpha p in k }}
   * alpha_max = frac{1}{sigma_x(p)}
   *
   * we want the largest alpha that stays within k for our next step
   * x_{k+1} = x_{k} - alpha p
   *
   * @param x in K
   * @param p in E
   *
   * @return alpha_max
   */
  constexpr T alpha_max(const Slack& x, const linalg::Vector<T, M>& p) const {
    return self().alpha_max_impl(x, p);
  }

  /**
   * @brief fraction to boundary dk^*
   *
   * sigma_x(p) = frac{1}{sup{alpha | y - alpha p in k^* }}
   * alpha_max = frac{1}{sigma_x(p)}
   *
   * we want the largest alpha that stays within k for our next step
   * y_{k+1} = y_{k} - alpha p
   *
   * @param y in K^*
   * @param p in E^*
   *
   * @return alpha_max
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
   * @param x in K
   *
   * @return F'(x)
   */
  constexpr auto grad(const Slack& x) const { return self().grad_impl(x); }

  /**
   * @brief e (jordan identity)
   *
   * x circ e = x
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
   * @param x in K
   * @param y in K^*
   *
   * @return (w, event)
   */
  constexpr auto w(const Slack& x, const Slack& y) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().w_impl(x, y);
  }

  /**
   * @brief circ (jordan map/product operator)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return x circ y
   */
  constexpr auto circ(const Slack& x, const Slack& y) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().circ_impl(x, y);
  }

  /**
   * @brief trace
   *
   * @param x in K
   * @param y in K^*
   *
   * @return tr(x circ y)
   */
  constexpr auto trace(const Slack& x, const Slack& y) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().trace_impl(x, y);
  }

  /**
   * @brief inv (jordan inverse element)
   *
   * @param x in K or y in K^*
   *
   * @return (x^{-1}, event)
   */
  constexpr auto inv(const Slack& x) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "Nonsymmetric cone is not equipped with Jordan Algebras");
    return self().inv_impl(x);
  }

  /**
   * @brief Convert Euclidean complementarity to this cone's Jordan trace units.
   *
   * For the supported symmetric cones, trace(u, v) = weight * <u, v>.
   * Evaluate that constant ratio at the Jordan identity: weight is 1 for
   * the orthant and PSD (svec coordinates), and 2 for SOC.
   *
   * The predictor-corrector uses mu_global = <x, y> / sum_k nu_k over the
   * entire product cone. Multiplying by this weight gives the local coefficient
   * of grad(x_k) = -inv(x_k) in its reduced complementarity residual.
   * This is a cone-specific normalization, independent of the current iterate;
   * it does not compute a local gap or change the shared centering target.
   */
  constexpr T trace_weight() const {
    const auto identity = e();
    return trace(identity, identity) /
           linalg::inner_product(identity, identity);
  }

  /**
   * @brief Affine second-order correction in dual coordinates under NT scaling.
   *
   * In the Euclidean Jordan algebra, let L_lambda(v) = lambda circ v.
   * With lambda = W^-1*x = W*y, solve
   * L_lambda(v) = (W^-1*px) circ (W*py), then return W^-1*v.
   * This converts the affine product to the reduced dual-step equation;
   * it is not a geometric cross product or an orthogonal-vector operation.
   *
   * Reference: L. Vandenberghe, "The CVXOPT linear and quadratic cone program
   * solvers," sections 4 (NT scaling) and 5, equation (19b).
   * https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
   */
  template <typename Scaling>
  constexpr auto scaled_affine_correction(const Slack& x, const Scaling& P,
                                          const Slack& px,
                                          const Slack& py) const {
    static_assert(ConeType::SYMMETRIC == CType,
                  "NT correction requires a symmetric cone");
    return self().scaled_affine_correction_impl(x, P, px, py);
  }

  /// Common symmetric-cone implementation using Jordan operations.
  template <typename Scaling>
  constexpr auto scaled_affine_correction_impl(const Slack& x, const Scaling& P,
                                               const Slack& px,
                                               const Slack& py) const {
    const auto [root, root_event] = self().sqrt_impl(P.x);
    if (linalg::is_fatal_event(root_event))
      return std::tuple(Slack{}, root_event);
    const auto [inv_root, inv_event] = inv(root);
    if (linalg::is_fatal_event(inv_event))
      return std::tuple(Slack{}, inv_event);
    const auto W_inv = Scaling(root);
    const auto W = Scaling(inv_root);
    const auto lambda = W_inv.apply(x);
    HessianMat jordan{};
    for (std::size_t j = 0; j < M; ++j) {
      Slack basis{};
      basis(j) = T(1);
      const auto column = circ(lambda, basis);
      for (std::size_t i = 0; i < M; ++i) jordan(i, j) = column(i);
    }
    const auto [correction, event] =
        linalg::LDL(jordan).solve(circ(W_inv.apply(px), W.apply(py)));
    if (linalg::is_fatal_event(event)) return std::tuple(Slack{}, event);
    return std::tuple(W_inv.apply(correction), event);
  }

  /**
   * @brief gamma_F (functional proximity neighborhood metric)
   *
   * @param x in K
   * @param y in K^*
   *
   * @return (gamma_F, event)
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

} // namespace conic
