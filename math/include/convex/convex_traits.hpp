/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/algebra_traits.hpp"

namespace convex {

/// @brief forward declaration of affine set
template <linalg::MatrixLike Mat, typename Derived>
class AffineSet;

/// @brief forward declaration of implicit affine set
template <linalg::MatrixLike Mat>
class ImplicitAffineSet;

/// @brief forward declaration of parametric affine set
template <linalg::MatrixLike Mat>
class ParametricAffineSet;

/// @brief forward declaration of convex set
template <linalg::VectorSpace Space, typename Derived>
class ConvexSet;

/// @brief forward declaration of halfspace
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace;

/// @brief forward declaration of norm ball set
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormBall;

/// @brief forward declaration of ellipsoid set
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           !linalg::is_complex_v<linalg::space_point_t<Space>>)
class Ellipsoid;

/// @brief forward declaration of norm cone set
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormCone;

/// @brief forward declaration of psd cone set
template <linalg::VectorSpace Space>
  requires(!linalg::is_complex_v<linalg::mat_type_t<Space>>)
class PsdCone;

/// @brief set traits
template <class Set>
struct set_traits;

/// @brief derived set type traits - affine sets
template <linalg::MatrixLike Mat, typename Derived>
struct set_traits<AffineSet<Mat, Derived>> {
  using value_type = typename linalg::mat_type_t<Mat>;
  using Point = linalg::Vector<value_type, linalg::mat_cols_v<Mat>>;
  static constexpr size_t Dim = linalg::mat_rows_v<Mat>;
};

/// @brief derived set type traits - implicit affine sets
template <linalg::MatrixLike Mat>
struct set_traits<ImplicitAffineSet<Mat>> {
  using value_type = typename linalg::mat_type_t<Mat>;
  using Point = linalg::Vector<value_type, linalg::mat_cols_v<Mat>>;
  static constexpr size_t Dim = linalg::mat_rows_v<Mat>;
};

/// @brief derived set type traits - parametric affine sets
template <linalg::MatrixLike Mat>
struct set_traits<ParametricAffineSet<Mat>> {
  using value_type = typename linalg::mat_type_t<Mat>;
  using Point = linalg::Vector<value_type, linalg::mat_cols_v<Mat>>;
  static constexpr size_t Dim = linalg::mat_rows_v<Mat>;
};

/// @brief derived set type traits - convex sets
template <linalg::VectorSpace Space, typename Derived>
struct set_traits<ConvexSet<Space, Derived>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traints - halfspaces
template <linalg::VectorSpace Space>
struct set_traits<Halfspace<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traints - norm ball
template <linalg::VectorSpace Space>
struct set_traits<NormBall<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traints - ellipsoid
template <linalg::VectorSpace Space>
struct set_traits<Ellipsoid<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traints - norm cone
template <linalg::VectorSpace Space>
struct set_traits<NormCone<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traints - psd cone
template <linalg::VectorSpace Space>
struct set_traits<PsdCone<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief set like concept
template <class Set>
concept SetLike =
    requires(const Set& set, const typename set_traits<Set>::Point& x) {
      { set.is_valid() } -> std::convertible_to<bool>;
      { set.contains(x) } -> std::convertible_to<bool>;
    };

}  // namespace convex
