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

/**
 * @brief Circ - Jordan Map/Product for SOC
 *
 * x \circ y = JxJ -> J
 * x \circ y = (x,t1) \circ (y,t2) = (x t2 + y t1, x^T y + t1 t2)
 *
 * @param x \in K
 * @param y \in K^*
 *
 * @return x \circ y (bilinear product)
 */
template <linalg::VectorLike Vec>
struct SOCCirc {
  /// @brief types and size extraction
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;

  /**
   * @brief circ (jordan map/product)
   *
   * x \circ y = (x,t1) \circ (y,t2) = (x t2 + y t1, x^T y + t1 t2)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return x \circ y (bilinear product)
   */
  constexpr auto operator()(const Vec& x, const Vec& y) const {
    const auto primal = std::get<0>(linalg::splice<N - 1>(x));
    const auto t1 = x(N - 1);
    const auto dual = std::get<0>(linalg::splice<N - 1>(y));
    const auto t2 = y(N - 1);

    const auto a = t2 * primal + t1 * dual;
    const auto b = linalg::dot(primal, dual) + t1 * t2;

    Vec res{};
    for (size_t i = 0; i < N - 1; ++i) res(i) = a(i);
    res(N - 1) = b;
    return res;
  }
};

/**
 * @brief soc cone class
 *
 * K_soc = { (x, t) | ||x|| <= t }
 */
template <linalg::VectorLike Slack>
struct SecondOrderCone
    : public Cone<Slack, SecondOrderCone<Slack>, ConeType::SYMMETRIC> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, SecondOrderCone<Slack>, ConeType::SYMMETRIC>;

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
    s(M - 1) = T(1);
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
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_cone_impl(const Slack& x) const {
    T res{};
    for (size_t i = 0; i < M - 1; ++i) res += x(i) * x(i);
    res = linalg::constexpr_sqrt(res);
    return res < x(M - 1);
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
   * sigma_x = \frac{1}{sup{alpha | x + alpha p in K}}
   * p = [ delta x; delta t ]
   * ||x + alpha delta(x)||_2^2 <= (t + alpha delta(t))^2
   * 0 <= (t + alpha delta(t))^2 - ||x + alpha delta(x)||_2^2
   * -inf(t^2 + 2 alpha t delta(t) + alpha^2 delta(t)^2 - x^2 - 2 alpha x
   * delta(x) - alpha^2 delta(x)^2)
   *
   * a alpha^2 + b alpha + c >= 0
   *
   * a = delta(t)^2 - delta(x)^2
   * b = 2 (t delta(t) - x^T delta(x))
   * c = t^2 - x^2 > 0
   *
   * alpha_max = piecewise{
   *  \frac{-b +- \sqrt(b^2 - 4 a c)}{2a}, a > 0,
   *  \frac{-c}{b}, a = 0, b > 0,
   *  +inf, a < 0
   * }
   *
   * @param x \in K
   * @param p \in E (delta dir)
   *
   * @return \alpha_max
   */
  constexpr T alpha_max_impl(const Slack& x,
                             const linalg::Vector<T, M>& p) const {
    constexpr T eta{0.99};
    T alpha{std::numeric_limits<T>::infinity()};
    T alpha_t{std::numeric_limits<T>::infinity()};

    const auto var = std::get<0>(linalg::splice<M - 1>(x));
    const auto t = x(M - 1);
    const auto dvar = std::get<0>(linalg::splice<M - 1>(p));
    const auto dt = p(M - 1);

    const auto a = dt * dt - linalg::dot(dvar, dvar);
    const auto b = T(2) * (t * dt - linalg::dot(var, dvar));
    const auto c = t * t - linalg::dot(var, var);

    if (linalg::constexpr_abs(a) <= eps)
      alpha = b < T(0) ? -c / b : alpha;
    else {
      const auto disc = b * b - T(4) * a * c;
      if (disc >= T(0)) {
        const auto s = linalg::constexpr_sqrt(disc);
        const auto r1 = (T(-1) * b - s) / (T(2) * a);
        const auto r2 = (T(-1) * b + s) / (T(2) * a);

        if (r1 > T(0)) alpha = std::min(alpha, r1);
        if (r2 > T(0)) alpha = std::min(alpha, r2);
      }
    }

    /// (dakre) ensure t + alpha dt > 0
    if (dt < T(0)) alpha_t = -t / dt;

    return eta * std::min(alpha, alpha_t);
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
   * F''(w) -> pushforward op same style as R^n_+
   *
   * w = P(x^{1/2})(P(x^{-1/2})y)^{1/2}
   *
   * let w = (x,t)
   * ** cone of squares **
   * x \circ x  = \hat{x} \circ \hat{x} = (2xt, x^T x + t^2); t^2 - x^T x >= 0
   * ** 2-eig / rank-2 jordan algebra **
   * two eigs: t +- x^T x
   * det(\hat{x}) = lam1*lam2 = t^2 - x^T x > 0
   * ** self-concordant barrier **
   * F(w) = F(x,t) = -log det(w)
   * F'(w) = -w^{-1}
   * F''(w) = P(w^{-1})
   *
   * let z = w^{-1}
   * P(z) = 2L(z)^2 - L(z^2)
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return H
   */
  constexpr auto H_impl(const Slack& x, const Slack& y) const {
    const auto w_inv = inv(w_impl(x, y));
    return linalg::QuadraticRep<HessianMat, SOCCirc<Slack>>(w_inv);
  }

  /**
   * @brief barrier param
   *
   * K = {(x,t) | t >= ||x||}
   * F(x,t) = -log(t^2 - ||x||^2)
   * t^2 - ||x||^2 -> lam1 = t + ||x||, lam2 = t - ||x||
   * det(x) = lam1*lam2 -> F = -log lam1 - log lam2
   *
   * ^ 2 log terms -> 2 barrier params
   *
   * @return nu
   */
  constexpr T nu_impl() const { return T(2); }

  /**
   * @brief e (jordan identity)
   *
   * e = (0,1)
   * (x,t) \circ e = (x,t)
   *
   * @return e
   */
  constexpr auto e_impl() const {
    Slack e{};
    e(M - 1) = T(1);
    return e;
  }

  /**
   * @brief w (self-scaled point)
   *
   * w = P(x^{1/2})(P(x^{-1/2})y)^{1/2}
   *
   * Euclidean Jordan Algebra
   * P(x) = 2L(x)^2 - L(x^2)
   * L(x)^2 = L(x) circ L(x)
   * L(x)y = x \circ y
   *
   * where u = (v / ||v||, 0)... primitive idempotent (directional term)
   *
   * x = lam1 c1 + lam2 c2... c1 & c2 are idempotents (rank-1 dir updates)
   * where x^{1/2} = \sqrt{lam1}c1 + \sqrt{lam2}c2
   *
   * let w = P(x^{1/2})z
   * let z = (P(x^{-1/2})y)^{1/2}
   *
   * z = sqrt(2 x^{-1/2} \circ (x^{-1/2} \circ y) - (x^{-1/2} \circ x^{-1/2})
   * \circ y)
   * w = 2 x^{1/2} \circ (x^{1/2} \circ z) - (x^{1/2} \circ x^{1/2}) \circ z
   *
   * @param x \in K
   * @param y \in K^*
   *
   * @return w
   */
  constexpr auto w_impl(const Slack& x, const Slack& y) const {
    const auto x_sqrt = sqrt(x);
    const auto x_sqrt_inv = inv(x_sqrt);

    SOCCirc<Slack> circ;

    const auto z = T(2) * circ(x_sqrt_inv, circ(x_sqrt_inv, y)) -
                   circ(circ(x_sqrt_inv, x_sqrt_inv), y);
    const auto z_sqrt = sqrt(z);

    const auto w = T(2) * circ(x_sqrt, circ(x_sqrt, z_sqrt)) -
                   circ(circ(x_sqrt, x_sqrt), z_sqrt);
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
    return SOCCirc<Slack>()(x, y);
  }

 private:
  /**
   * @brief sqrt
   *
   * Jordan Square Root
   * lam1 = t + ||v||_2, lam2 = t - ||v||_2
   * x^{1/2} = \frac{sqrt(lam1) + sqrt(lam2)}{2}e + \frac{sqrt(lam1) -
   * sqrt(lam2)}{2}u
   *
   * @param x (slack vector)
   *
   * @return sqrt(x)
   */
  constexpr auto sqrt(const Slack& x) const {
    const auto v = std::get<0>(linalg::splice<M - 1>(x));
    const auto t = x(M - 1);

    const auto v_norm = linalg::L2<decltype(v)>()(v);
    const auto lam1 = t + v_norm;
    const auto lam2 = t - v_norm;

    const auto alpha =
        T(0.5) * (linalg::constexpr_sqrt(lam1) + linalg::constexpr_sqrt(lam2));
    const auto beta =
        T(0.5) * (linalg::constexpr_sqrt(lam1) - linalg::constexpr_sqrt(lam2));

    Slack v_zero{};
    for (size_t i = 0; i < M - 1; ++i) v_zero(i) = v(i);

    auto x_sqrt = alpha * e_impl();
    if (v_norm > eps) x_sqrt = x_sqrt + beta * (T(1) / v_norm) * v_zero;

    return x_sqrt;
  }

  /**
   * @brief inv
   *
   * Jordan Inverse of x where x = (v,t)
   * x^{-1} = \frac{1}{t^2 - v^Tv}(-v,t)... \frac{1}{det(A)}*adj(A)
   *
   * @param x (slack vector)
   *
   * @return inv(x)
   */
  constexpr auto inv(const Slack& x) const {
    const auto v = std::get<0>(linalg::splice<M - 1>(x));
    const auto t = x(M - 1);
    Slack adj_v;
    for (size_t i = 0; i < M - 1; ++i) adj_v(i) = T(-1) * v(i);
    adj_v(M - 1) = t;
    auto div = t * t - linalg::dot(v, v);
    if (linalg::constexpr_abs(div) < eps) div = eps;
    return (T(1) / div) * adj_v;
  }
};

}  // namespace conic
