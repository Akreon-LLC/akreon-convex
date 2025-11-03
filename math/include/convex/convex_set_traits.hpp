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

#include "affine/affine_traits.hpp"
#include "algebra/algebra_traits.hpp"

namespace convex {

/// @brief forward declaration of projective map
template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  requires(std::same_as<linalg::mat_type_t<MatA>, linalg::mat_type_t<MatC>> &&
           linalg::mat_cols_v<MatA> == linalg::mat_cols_v<MatC> &&
           linalg::mat_rows_v<MatC> == 1)
class ProjectiveMap;

/// @brief forward declaration of convex set
template <linalg::RnSpace Space, typename Derived>
class ConvexSet;

/// @brief convex set view tag
enum struct ViewTag { ConvexSet = 0, LiftedCone, EpigraphCone, HypographCone };

/// @brief simple enum to describe type of halfspace
enum struct HalfspaceType { CLH, OLH, CUH, OUH };

/// @brief simple enum to describe type of ellipsoid
enum struct EllipsoidType { CE, OE };

/// @brief simple enum to describe ball type
enum struct NormBallType { L2OB, L2CB, L1OB, L1CB, LinfOB, LinfCB };

/// @brief simple enum for cone type
enum struct NormConeType { L2OC, L2CC, L1OC, L1CC, LinfOC, LinfCC };

/// @brief LMIType
enum struct LMIType { Strict, NonStrict };

/// @brief forward declaration of hyperplane
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Hyperplane;

/// @brief forward declaration of halfspace
template <linalg::RnSpace Space, HalfspaceType Type, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace;

/// @brief forward declaration of norm ball set
template <linalg::RnSpace Space, NormBallType Type, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           Tag == ViewTag::ConvexSet)
class NormBall;

/// @brief forward declaration of ellipsoid set
template <linalg::RnSpace Space, EllipsoidType Type, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Ellipsoid;

/// @brief forward declaration of norm cone set
template <linalg::RnSpace Space, NormConeType Type, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormCone;

/// @brief forward declaration of psd cone set
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Space>>)
class PsdCone;

/// @brief forward declaration of polyhedron set
template <linalg::RnSpace Space, HalfspaceType Type, linalg::MatrixLike MatA,
          linalg::MatrixLike MatC, ViewTag Tag>
  requires(!linalg::is_complex_v<linalg::mat_type_t<Space>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatA>> &&
           !linalg::is_complex_v<linalg::mat_type_t<MatC>>)
class Polyhedron;

/// @brief forward declaration of simplex set
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Simplex;

/// @brief forward declaration of nonneg orthant
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NonnegOrthant;

/// @brief forward declaration of LMI type
template <linalg::RnSpace Space, LMIType Type, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class LMI;

/// @brief rotated second order cone type
template <linalg::RnSpace Space, NormConeType Type, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> > 1 &&
           (Type == NormConeType::L2OC || Type == NormConeType::L2CC))
class RotatedSOC;

/// @brief exponential cone type
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> == 2)
class ExpCone;

/// @brief log det cone type
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Space>>)
class LogDetCone;

/// @brief power cone type
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class PowerCone;

/// @brief log sum exp cone type
template <linalg::RnSpace Space, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class LogSumExpCone;

/// @brief affine subspace type
template <linalg::RnSpace Space, linalg::MatrixLike Mat, ViewTag Tag>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           linalg::space_dim_v<Space> == linalg::mat_cols_v<Mat>)
class AffineSubspace;

/// @brief set traits
template <class Set>
struct set_traits;

/// @brief derived set type traits - R^n
template <linalg::RnSpace Space>
struct set_traits<Space> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - convex sets
template <linalg::RnSpace Space, typename Derived>
struct set_traits<ConvexSet<Space, Derived>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
};

/// @brief derived set type traits - hyperplane
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<Hyperplane<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - halfspaces
template <linalg::RnSpace Space, HalfspaceType Type, ViewTag Tag>
struct set_traits<Halfspace<Space, Type, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - norm ball
template <linalg::RnSpace Space, NormBallType Type, ViewTag Tag>
struct set_traits<NormBall<Space, Type, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - ellipsoid
template <linalg::RnSpace Space, EllipsoidType Type, ViewTag Tag>
struct set_traits<Ellipsoid<Space, Type, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - norm cone
template <linalg::RnSpace Space, NormConeType Type, ViewTag Tag>
struct set_traits<NormCone<Space, Type, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - psd cone
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<PsdCone<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - polyhedron
template <linalg::RnSpace Space, HalfspaceType Type, linalg::MatrixLike MatA,
          linalg::MatrixLike MatC, ViewTag Tag>
struct set_traits<Polyhedron<Space, Type, MatA, MatC, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - simplex
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<Simplex<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - nonneg orthant
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<NonnegOrthant<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - LMI
template <linalg::RnSpace Space, LMIType Type, ViewTag Tag>
struct set_traits<LMI<Space, Type, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - RSOC
template <linalg::RnSpace Space, NormConeType Type, ViewTag Tag>
struct set_traits<RotatedSOC<Space, Type, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - Exp cone
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<ExpCone<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - Log det cone
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<LogDetCone<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - power cone
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<PowerCone<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - log sum exp
template <linalg::RnSpace Space, ViewTag Tag>
struct set_traits<LogSumExpCone<Space, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief derived set type traits - log sum exp
template <linalg::RnSpace Space, linalg::MatrixLike Mat, ViewTag Tag>
struct set_traits<AffineSubspace<Space, Mat, Tag>> {
  using RnSpace = Space;
  using value_type = typename linalg::space_type_t<Space>;
  using Point = typename linalg::space_point_t<Space>;
  static constexpr size_t DIM = linalg::space_dim_v<Space>;
  static constexpr ViewTag VIEW_TAG = Tag;
};

/// @brief alias for vector space
template <class Set>
using set_space_t = typename set_traits<linalg::remove_cvref_t<Set>>::RnSpace;

/// @brief alias for set type
template <class Set>
using set_type_t = typename set_traits<linalg::remove_cvref_t<Set>>::value_type;

/// @brief alias for point type in set
template <class Set>
using set_point_t = typename set_traits<linalg::remove_cvref_t<Set>>::Point;

/// @brief helper to get convex set view tag
template <class Set>
inline constexpr ViewTag set_view_tag_v = set_traits<Set>::VIEW_TAG;

/// @brief helper to get dimension of space underpinning set
template <class Set>
inline constexpr size_t set_dim_v =
    set_traits<linalg::remove_cvref_t<Set>>::DIM;

/// @brief set like concept
template <class Set>
concept SetLike = requires(const Set& set, const set_point_t<Set>& x) {
  { set.is_well_formed() } -> std::convertible_to<bool>;
  { set.contains(x) } -> std::convertible_to<bool>;
};

/// @brief lifted set (x,t) like concept
template <class Set>
concept LiftedSetLike = requires(
    const Set& set, const std::tuple<set_point_t<Set>, set_type_t<Set>>& x) {
  { set.is_well_formed() } -> std::convertible_to<bool>;
  { set.contains(x) } -> std::convertible_to<bool>;
};

/// @brief convex set like concept
template <class Set>
concept ConvexSetLike =
    (SetLike<Set> && set_view_tag_v<Set> == ViewTag::ConvexSet) ||
    linalg::RnSpace<Set>;

/// @brief cone set like concept
template <class Set>
concept ConvexConeLike =
    LiftedSetLike<Set> && (set_view_tag_v<Set> == ViewTag::LiftedCone ||
                           set_view_tag_v<Set> == ViewTag::EpigraphCone ||
                           set_view_tag_v<Set> == ViewTag::HypographCone);

}  // namespace convex
