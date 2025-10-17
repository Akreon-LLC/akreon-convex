/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "cone_traits.hpp"

namespace convex {

template <typename Derived>
struct ConeViewBase {
  /// @brief alias to extract underlying scalar type of convex set
  using T = cone_view_type_t<Derived>;

  /// @brief alias for point
  using Point = cone_view_point_t<Derived>;

  /// @brief alias for space underlying cone
  using Space = cone_view_space_t<Derived>;

  /// @brief alias for cone in view
  using Cone = cone_view_cone_t<Derived>;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit ConeViewBase(const Cone& cone_in) : cone(cone_in) {}

  /**
   * @brief contains func
   *
   * @param point in space
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const Point& x, T tol = 1e-8) const {
    return self().contains_impl(x, tol);
  }

  /**
   * @brief is proper check
   *
   * 1. K is convex (satisfied)
   * 2. K is closed (limit (x, t) \in K included)
   * 3. K is solid
   * 4. K is pointed x \in K -x \notin K except for 0 (K \intersect -K = 0)
   *
   * @return { False, True, Unknown }
   */
  constexpr ConeCheck is_proper() const {
    cone_traits<Cone> ct;
    const auto closed = ct.is_closed(cone);
    if (closed == ConeCheck::False) return ConeCheck::False;
    const auto solid = ct.is_solid(cone);
    if (solid == ConeCheck::False) return ConeCheck::False;
    const auto pointed = ct.is_pointed(cone);
    if (pointed == ConeCheck::False) return ConeCheck::False;

    if (closed == ConeCheck::Unknown || solid == ConeCheck::Unknown ||
        pointed == ConeCheck::Unknown)
      return ConeCheck::Unknown;

    return ConeCheck::True;
  }

  /**
   * @brief general inequality for cone K (must be proper)
   *
   * x <=_K y
   *
   * @param Point x \in Space
   * @param Point y \in Space
   *
   * @return T/F
   */
  constexpr std::optional<bool> less_than(const Point& x, const Point& y,
                                          T tol = 1e-8) const {
    if (is_proper() == ConeCheck::False) return std::nullopt;
    return contains(y - x, tol);
  }

  /**
   * @brief strict inequality for cone K (must be proper)
   *
   * x <_K y
   *
   * @param Point x \in Space
   * @param Point y \in Space
   *
   * @return T/F
   */
  constexpr std::optional<bool> less_than_strict(const Point& x, const Point& y,
                                                 T tol = 1e-8) const {
    if (is_proper() == ConeCheck::False) return std::nullopt;

    const auto delta = y - x;
    const auto eps = 2 * tol;
    if constexpr (std::same_as<Point, linalg::Matrix<T, N, N>>) {
      linalg::Matrix<T, N, N> res;
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i)(j) - x(i)(j) + eps;

      if (!contains(res, tol)) return false;

      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i)(j) - x(i)(j) - eps;

      if (!contains(res, tol)) return false;
    } else {
      const linalg::vec_traits<linalg::Vector<T, N>> vec;
      const auto wiggle = vec.fill(eps);
      if (!contains(delta + wiggle, tol)) return false;
      if (!contains(delta - wiggle, tol)) return false;
    }
    return true;
  }

  /**
   * @brief get dual view of cone
   *
   * @return dual cone view
   */
  constexpr auto dual_view() const { return self().dual_view_impl(cone); }

  /**
   * @brief self func to return dervied object instance
   *
   * return self
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }

  /** @brief member vars for cone */
  Cone cone;  //< Cone representation of underlying convex set
};

}  // namespace convex
