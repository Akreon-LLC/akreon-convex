/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <iostream>

#include "convex_set.hpp"

namespace convex {

/**
 * @brief exponential cone
 *
 * { (x, y, z) | y e^{x/y} <= z, y > 0 } \cup { (x, 0, z) | x <= 0, z >= 0 }
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> == 3)
class ExpCone : public ConvexSet<Space, ExpCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, ExpCone<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief exp cone \subset R^3
   *
   * @param Point \in R^3
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /* tol */) const {
    if (x(0) > T(0) || x(1) < T(0) || x(2) < T(0)) return false;
    if (x(1) == T(0)) return true;
    const auto res = x(1) * linalg::constexpr_exp(x(0) / x(1));
    return res <= x(2);
  }

  /**
   * @brief is well formed check
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief is equal impl
   */
  constexpr bool is_equal_impl(const ConvexBase& /* other */) const {
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
    requires(Tag == ViewTag::ConvexSet)
  constexpr auto transform_impl(const AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const
    requires(Tag == ViewTag::ConvexSet)
  {
    return ConeView(ExpCone<Space, ViewTag::ConvexCone>());
  }

  /**
   * @brief project impl
   *
   * @param project map
   * @param in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
    requires(Tag == ViewTag::ConvexSet)
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  project_impl(const ProjectiveMap<MatA, MatC>& /* map */,
               const Point& /* x */) const {
    return std::nullopt;
  }
};

}  // namespace convex
