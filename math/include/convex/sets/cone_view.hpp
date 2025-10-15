/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../convex_traits.hpp"

namespace convex {

/**
 * @brief this is an owning cone view of the underlying convex set
 */
template <ConvexConeLike Cone>
class ConeView {
 public:
  /// @brief alias to extract underlying scalar type of convex set
  using T = set_type_t<Cone>;

  /// @brief alias for point
  using Point = set_point_t<Cone>;

  /// @brief alias for space underlying cone
  using Space = set_space_t<Cone>;

  /// @brief helper to get dimension of space
  static constexpr size_t Dim = linalg::space_dim_v<Space>;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<linalg::space_point_t<Space>>;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit ConeView(const Cone& cone_in) : cone(cone_in) {}

  /**
   * @brief contains func (wrapped around cone impl)
   *
   * @param point in space
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const Point& x, T tol = 1e-8) const {
    return cone.contains_impl(x, tol);
  }

  /**
   * @brief is proper check
   *
   * 1. K is convex (satisfied)
   * 2. K is closed (limit (x, t) \in K included)
   * 3. K is solid
   * 4. K is pointed x \in K -x \notin K except for 0 (K \intersect -K = 0)
   *
   * @return T/F
   */
  constexpr bool is_proper() const {
    // TODO - add some cone traits to each convex set to enable is proper check
    // to run smoothly

    // verify non-empty interior = solid
    if (!cone.is_well_formed_impl() || Dim == 0) return false;

    return true;
  }

  /**
   * @brief general inequality for cone K
   *
   * x <=_K y
   *
   * @param Point x \in Space
   * @param Point y \in Space
   *
   * @return T/F
   */
  constexpr bool lte(const Point& x, const Point& y, T tol = 1e-8) const {
    if constexpr (std::same_as<Point, linalg::SymmetricMatrix<
                                          linalg::Matrix<T, N, N>>>) {
      linalg::Matrix<T, N, N> res;
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i)(j) - x(i)(j);

      linalg::SymmetricMatrix<decltype(res)> S(res);
      return S.is_psd();
    } else {
      return contains(y - x, tol);
    }
  }

  /**
   * @brief strict inequality for cone K
   *
   * x <_K y
   *
   * @param Point x \in Space
   * @param Point y \in Space
   *
   * @return T/F
   */
  constexpr bool lt(const Point& x, const Point& y, T tol = 1e-8) const {
    const auto delta = y - x;
    const auto eps = 2 * tol;
    if constexpr (std::same_as<Point, linalg::SymmetricMatrix<
                                          linalg::Matrix<T, N, N>>>) {
      linalg::Matrix<T, N, N> res;
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i, j) - x(i, j) + eps;

      const linalg::SymmetricMatrix<decltype(res)> Sp(res);
      if (!Sp.is_psd()) return false;

      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i, j) - x(i, j) - eps;

      const linalg::SymmetricMatrix<decltype(res)> Sn(res);
      if (!Sn.is_psd()) return false;
    } else {
      const linalg::vec_traits<linalg::Vector<T, N>> vec;
      const auto wiggle = vec.fill(eps);
      if (!contains(delta + wiggle, tol)) return false;
      if (!contains(delta - wiggle, tol)) return false;
    }
    return true;
  }

 private:
  Cone cone;
};

}  // namespace convex
