/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/algebra_traits.hpp"

namespace convex {

/// @brief forward declaration of affine map
template <linalg::MatrixLike Mat>
class AffineMap;

/// @brief forward declaration of projective map
template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  requires(std::same_as<linalg::mat_type_t<MatA>, linalg::mat_type_t<MatC>> &&
           linalg::mat_cols_v<MatA> == linalg::mat_cols_v<MatC> &&
           linalg::mat_rows_v<MatC> == 1)
class ProjectiveMap;

/// @brief forward declaration of affine set
template <linalg::MatrixLike Mat, typename Derived>
class AffineSet;

/// @brief forward declaration of parametric affine set
template <linalg::MatrixLike Mat>
class ParametricAffineSet;

/// @brief forward declaration of convex set
template <linalg::VectorSpace Space, typename Derived>
  requires(!linalg::is_complex_v<linalg::space_type_t<Space>>)
class ConvexSet;

/// @brief forward declaration of halfspace type
enum struct HalfspaceType;

/// @brief forward declaration of ellipse type
enum struct EllipsoidType;

/// @brief forward declaration of norm ball type
enum struct NormBallType;

/// @brief forward declaration of norm ball type
enum struct NormConeType;

/// @brief forward declaration of simplex type
enum struct SimplexType;

/// @brief forward declaration of hyperplane
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Hyperplane;

/// @brief forward declaration of halfspace
template <linalg::VectorSpace Space, HalfspaceType Type>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace;

/// @brief forward declaration of norm ball set
template <linalg::VectorSpace Space, NormBallType Type>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormBall;

/// @brief forward declaration of ellipsoid set
template <linalg::VectorSpace Space, EllipsoidType Type>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           !linalg::is_complex_v<linalg::space_point_t<Space>>)
class Ellipsoid;

/// @brief forward declaration of norm cone set
template <linalg::VectorSpace Space, NormConeType Type>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormCone;

/// @brief forward declaration of psd cone set
template <linalg::VectorSpace Space>
  requires(!linalg::is_complex_v<linalg::mat_type_t<Space>>)
class PsdCone;

/// @brief forward declaration of polyhedron set
template <linalg::VectorSpace Space, HalfspaceType Type,
          linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  requires(!linalg::is_complex_v<linalg::mat_type_t<Space>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatA>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatC>>)
class Polyhedron;

/// @brief forward declaration of simplex set
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           !linalg::is_complex_v<linalg::space_point_t<Space>>)
class Simplex;

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

/// @brief derived set type traits - parametric affine sets
template <linalg::MatrixLike Mat>
struct set_traits<ParametricAffineSet<Mat>> {
  using value_type = typename linalg::mat_type_t<Mat>;
  using Point = linalg::Vector<value_type, linalg::mat_cols_v<Mat>>;
  static constexpr size_t Dim = linalg::mat_rows_v<Mat>;
};

/// @brief derived set type traits - convex sets
template <linalg::VectorSpace Space, typename Derived>
  requires(!linalg::is_complex_v<linalg::space_type_t<Space>>)
struct set_traits<ConvexSet<Space, Derived>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - hyperplane
template <linalg::VectorSpace Space>
struct set_traits<Hyperplane<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - halfspaces
template <linalg::VectorSpace Space, HalfspaceType Type>
struct set_traits<Halfspace<Space, Type>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - norm ball
template <linalg::VectorSpace Space, NormBallType Type>
struct set_traits<NormBall<Space, Type>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - ellipsoid
template <linalg::VectorSpace Space, EllipsoidType Type>
struct set_traits<Ellipsoid<Space, Type>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - norm cone
template <linalg::VectorSpace Space, NormConeType Type>
struct set_traits<NormCone<Space, Type>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - psd cone
template <linalg::VectorSpace Space>
struct set_traits<PsdCone<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - polyhedron
template <linalg::VectorSpace Space, HalfspaceType Type,
          linalg::MatrixLike MatA, linalg::MatrixLike MatC>
struct set_traits<Polyhedron<Space, Type, MatA, MatC>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - simplex
template <linalg::VectorSpace Space>
struct set_traits<Simplex<Space>> {
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
};

/// @brief set like concept
template <class Set>
concept SetLike =
    requires(const Set& set, const typename set_traits<Set>::Point& x) {
      { set.is_well_formed() } -> std::convertible_to<bool>;
      { set.contains(x) } -> std::convertible_to<bool>;
    };

}  // namespace convex
