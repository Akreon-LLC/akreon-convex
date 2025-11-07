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

#include "convex_set.hpp"

namespace convex {

/**
 * @brief default coord map for power cone
 *
 * coords: (x, t)
 */
template <linalg::SquareMatrixLike CoordMat>
constexpr auto default_pow_cone_map(const linalg::mat_type_t<CoordMat>& t) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;
  auto I = linalg::make_identity<T, N>();
  I[N - 1][N - 1] = t;
  return affine::AffineMap<CoordMat>(I, linalg::Vector<T, N>{});
}

/**
 * @brief power cone
 *
 * Symmetric Power Cone (default)
 * { (x, y) \in R^n_+ x R | |y| <= x_1^\alpha_1 * x_2^\alpha_2 * ... , \sum_i^n
 * \alpha_i <= 1, \forall \alpha >= 0 }
 *
 * (dakre) TODO general lifted cone doesnt have abs on t ... need to evaluate
 * efficacy of this approach
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::LiftedCone>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class PowerCone : public ConvexSet<Space, PowerCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, PowerCone<Space, Tag>>;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief default power cone CTOR
   */
  constexpr explicit PowerCone()
      : ConvexBase(default_pow_cone_map<CoordMat>(T(1))) {
    for (size_t i = 0; i < N; ++i) roots[i] = N;
  }

  /**
   * @brief default power cone CTOR
   */
  constexpr explicit PowerCone(const affine::AffineMap<CoordMat>& map_in)
      : ConvexBase(map_in) {
    for (size_t i = 0; i < N; ++i) roots[i] = N;
  }

  /**
   * @brief power cone (roots provided)
   *
   * @param roots \in N
   */
  constexpr explicit PowerCone(const linalg::Vector<size_t, N>& roots_in)
      : ConvexBase(default_pow_cone_map<CoordMat>(T(1))), roots(roots_in) {}

  /**
   * @brief power cone (roots provided)
   *
   * @param affine map (coord map)
   * @param roots \in N
   */
  constexpr explicit PowerCone(const affine::AffineMap<CoordMat>& map_in,
                               const linalg::Vector<size_t, N>& roots_in)
      : ConvexBase(map_in), roots(roots_in) {}

  /**
   * @brief power cone contains \subset R^n_+ x R
   *
   * @param x \in R^n_+ x R
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::tuple(x, T(1)), tol);
  }

  /**
   * @brief power cone contains \subset R^n_+ x R
   *
   * Power Cone
   * { (x, t) \in R^n_+ x R | \prod_i^n x_i^{1/n} >= t }
   *
   * @param (x, t) \in R^n_+ x R
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x_proj = this->project(x_in);
    const auto x = std::get<0>(x_proj);
    const auto s = std::get<1>(x_proj);

    T prod(1);
    for (size_t i = 0; i < N; ++i)
      prod *= linalg::constexpr_nrt<T>(x(i), roots(i));

    return prod >= s;
  }

  /**
   * @brief is well formed check
   *
   * @param tolerance scalar
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl(T tol = 1e-8) const {
    T sum(0);
    std::for_each(roots.begin(), roots.end(),
                  [&](const auto& root) { sum += T(1) / T(root); });
    return linalg::constexpr_abs(sum - T(1)) < tol;
  }

  /**
   * @brief is equal impl
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    if (other.coord_map != this->coord_map) return false;

    for (size_t i = 0; i < N; ++i)
      if (other.self().roots(i) != roots(i)) return false;
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(
          PowerCone<Space, ViewTag::LiftedCone>(this->coord_map, roots));
    else
      return ConeView(PowerCone<Space, Tag>(this->coord_map, roots));
  }

 private:
  /** @brief member vars */
  linalg::Vector<size_t, N> roots;  //< alphas encoded as roots
};

}  // namespace convex
