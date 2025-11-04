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

#include "convex_set_traits.hpp"

namespace convex {

/// @brief forward declaration of cone view
template <ConvexConeLike Cone>
struct ConeView;

/// @brief forward declaration of dual cone view
template <ConvexConeLike Cone>
struct DualConeView;

/// @brief cone view
template <class Cone>
struct cone_view_traits;

/// @brief derived for cone view
template <ConvexConeLike Cone>
struct cone_view_traits<ConeView<Cone>> {
  using T = set_type_t<Cone>;
  using Point = set_point_t<Cone>;
  using Space = set_space_t<Cone>;
  using K = Cone;
};

/// @brief derived for dual cone view
template <ConvexConeLike Cone>
struct cone_view_traits<DualConeView<Cone>> {
  using T = set_type_t<Cone>;
  using Point = set_point_t<Cone>;
  using Space = set_space_t<Cone>;
  using K = Cone;
};

/// @brief helper to get type in view
template <class Cone>
using cone_view_type_t =
    typename cone_view_traits<linalg::remove_cvref_t<Cone>>::T;

/// @brief helper to get point type in view
template <class Cone>
using cone_view_point_t =
    typename cone_view_traits<linalg::remove_cvref_t<Cone>>::Point;

/// @brief helper to get space type in view
template <class Cone>
using cone_view_space_t =
    typename cone_view_traits<linalg::remove_cvref_t<Cone>>::Space;

/// @brief helper to get cone K in view
template <class Cone>
using cone_view_cone_t =
    typename cone_view_traits<linalg::remove_cvref_t<Cone>>::K;

/// @brief helper to get dim of point in cone
template <class Cone>
inline constexpr size_t cone_view_dim_v =
    linalg::vec_size_v<cone_view_point_t<linalg::remove_cvref_t<Cone>>>;

/// @brief cone traits
template <class Cone>
struct cone_traits;

/// @brief cone check enum
enum struct ConeCheck { False = 0, True, Unknown };

/// @brief cone view concept
template <typename Cone>
concept ConeViewLike = requires(
    const Cone& cone,
    const std::tuple<cone_view_point_t<Cone>, cone_view_type_t<Cone>>& x,
    const affine::AffineMap<linalg::Matrix<
        cone_view_type_t<Cone>, cone_view_dim_v<Cone>, cone_view_dim_v<Cone>>>&
        map) {
  { cone.is_proper() } -> std::convertible_to<ConeCheck>;
  { cone.contains(x) } -> std::convertible_to<bool>;
  { cone.compute_slack(map, x) } -> std::convertible_to<decltype(x)>;
  { cone.dual_view() };
};

/// @brief derived cone type traits - lifted hyperplane
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<Hyperplane<Space, Tag>> {
  using ConeT = Hyperplane<Space, Tag>;
  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::False;
  }
};

/// @brief derived cone type traits - lifted halfspace
template <linalg::RnSpace Space, HalfspaceType Type, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<Halfspace<Space, Type, Tag>> {
  using ConeT = Halfspace<Space, Type, Tag>;

  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return Type == HalfspaceType::CLH || Type == HalfspaceType::CUH
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::False;
  }
};

/// @brief derived cone type traits - lifted ellipsoid
template <linalg::RnSpace Space, EllipsoidType Type, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<Ellipsoid<Space, Type, Tag>> {
  using ConeT = Ellipsoid<Space, Type, Tag>;

  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return Type == EllipsoidType::CE ? ConeCheck::True : ConeCheck::False;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::True
               : ConeCheck::False;
  }

  /**
   * @brief pointed test (x,t) \in K & (-x, -t) \in -K
   *
   * (i) (x - ct)^T P^{-1} (x - ct) <= t^2
   * (ii) (-x + ct)^T P{-1} (-x + ct) <= (-t)^2 = t^2
   *
   * both inequalities are bounded by t^2 & (x - ct)^2 \equiv (-x + ct)^2 so
   * their relative magnitudes are the same (scaled by same P)
   *
   * -t < 0 and hence not in -K
   *
   *  K \cap -K = {0}
   */
  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }
};

/// @brief derived cone type traits - norm cone
template <linalg::RnSpace Space, NormConeType Type, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<NormCone<Space, Type, Tag>> {
  using ConeT = NormCone<Space, Type, Tag>;
  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return Type == NormConeType::L2CC || Type == NormConeType::L1CC ||
                   Type == NormConeType::LinfCC
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::True
               : ConeCheck::False;
  }

  /**
   * @brief pointed test (x,t) \in K & (-x, -t) \in -K
   *
   * (i) { (x, t) | ||x|| <= t }
   * (ii) { (-x, -t) | ||-x|| <= -t }
   *
   * -t < 0 thus not in -K
   *
   * L2 case : x \in K, -x \in -K but -t invalid so K \cap -K = {0}
   * L1 case : similar result
   * Linf case : similar result
   */
  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }
};

/// @brief derived cone type traits for lifted simplex
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<Simplex<Space, Tag>> {
  using ConeT = Simplex<Space, Tag>;
  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::True
               : ConeCheck::False;
  }

  /**
   * @brief pointed test (x,t) \in K & (-x, -t) \in -K
   *
   * (i) { (x, t) | \sum_i x_i = t, t > 0 }
   * (ii) { (x, t) | \sum_i -x_i = -t, -t > 0 }
   *
   * set 2 cannot exist thus K \cap -K = { 0 }
   */
  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }
};

/// @brief derived cone type traits for nonneg orthant
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<NonnegOrthant<Space, Tag>> {
  using ConeT = NonnegOrthant<Space, Tag>;
  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }
};

/// @brief psd cone type traits
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<PsdCone<Space, Tag>> {
  using ConeT = PsdCone<Space, Tag>;
  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::mat_rows_v<Space> > 1 &&
                   linalg::mat_cols_v<Space> > 1
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }
};

/// @brief polyhedron cone type traits
template <linalg::RnSpace Space, HalfspaceType Type, linalg::MatrixLike MatA,
          linalg::MatrixLike MatC, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<Polyhedron<Space, Type, MatA, MatC, Tag>> {
  using ConeT = Polyhedron<Space, Type, MatA, MatC, Tag>;
  using T = linalg::space_type_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;

  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return Type == HalfspaceType::CLH || Type == HalfspaceType::CUH
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    // (dakre) this is a soft check (full solid check includes slater test for
    // feasibility (LP))
    return cone.is_well_formed() && N > 0 ? ConeCheck::Unknown
                                          : ConeCheck::False;
  }

  /**
   * @brief pointed test
   *
   * (i) { (x, t) | A x <= t, C x = d t }
   * (ii) { (-x, -t) | -A x <= -t, -C x = -d t }
   *
   * t > 0 by def, so t = 0 and rhs \to 0
   *
   * K \cap -K = { 0 } \implies L(P) = nullspace([A C]) AKA lineality space
   * IFF L(P) = null([A C]) = { 0 } is trivial then it's pointed
   */
  static constexpr ConeCheck is_pointed(const ConeT& cone) {
    const auto A = cone.get_A();
    static constexpr size_t RowsC = linalg::mat_rows_v<MatC>;
    static constexpr size_t RowsA = linalg::mat_rows_v<MatA>;
    if constexpr (RowsC == 0) {
      const auto rank = linalg::QR(A).get_rank();
      return rank == N ? ConeCheck::True : ConeCheck::False;
    } else {
      const auto C = cone.get_C();
      linalg::Matrix<T, RowsA + RowsC, N> AC;
      AC = linalg::set_block<0, RowsA, 0, N>(AC, A);
      AC = linalg::set_block<RowsA, RowsA + RowsC, 0, N>(AC, C);
      const auto rank = linalg::QR(AC).get_rank();
      return rank == N ? ConeCheck::True : ConeCheck::False;
    }
  }
};

/// @brief LMI cone type traits
template <linalg::RnSpace Space, LMIType Type, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<LMI<Space, Type, Tag>> {
  using ConeT = LMI<Space, Type, Tag>;
  using T = linalg::space_type_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;

  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return Type == LMIType::NonStrict ? ConeCheck::True : ConeCheck::False;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    // (dakre) this is a soft check (full solid check includes slater test for
    // feasibility (LP))
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::Unknown
               : ConeCheck::False;
  }

  /**
   * @brief pointed test
   *
   * (i) { (x, t) | A(x) <= Bt }
   * (ii) { (-x, -t) | A(-x) <= -Bt }
   *
   * 0 <= Bt - A(x) \implies PSD
   * 0 <= A(x) - Bt \implies PSD
   *
   * t > 0 by def, so t = 0 and B drops out
   * 0 <= A(x) & 0 <= -A(x) \implies A(x) <= 0
   * IFF null(A(x)) = { 0 } is trivial then its pointed
   */
  static constexpr ConeCheck is_pointed(const ConeT& cone) {
    const auto As = cone.get_A();
    linalg::Matrix<T, N * N, N> A_stack;
    linalg::constexpr_for<0, N>(
        [&]<size_t i>(std::integral_constant<size_t, i>) {
          constexpr size_t blk = i * N;
          A_stack = linalg::set_block<blk, N + blk, 0, N>(A_stack, As(i));
        });
    const auto rank = linalg::QR(A_stack).get_rank();
    return rank == N ? ConeCheck::True : ConeCheck::False;
  }
};

/// @brief Rotated SOC
template <linalg::RnSpace Space, NormConeType Type, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<RotatedSOC<Space, Type, Tag>> {
  using ConeT = RotatedSOC<Space, Type, Tag>;

  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return Type == NormConeType::L2CC ? ConeCheck::True : ConeCheck::False;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() && linalg::space_dim_v<Space> > 0
               ? ConeCheck::True
               : ConeCheck::False;
  }

  static constexpr ConeCheck is_pointed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }
};

/// @brief exponential cone
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<ExpCone<Space, Tag>> {
  using ConeT = ExpCone<Space, Tag>;
  using T = linalg::space_type_t<Space>;

  static constexpr ConeCheck is_closed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() ? ConeCheck::True : ConeCheck::False;
  }

  /**
   * @brief pointed test
   *
   * (i) { (x, y, z) | y e^{x/y} >= z, x <= 0, z >= 0, y > 0 } \cup { (x, 0, z)
   * | x <= 0, z >= 0 } (ii) -K = { 0 }
   *
   * so K \cap -K = { 0 }
   */
  static constexpr ConeCheck is_pointed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }
};

/// @brief log det cone
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<LogDetCone<Space, Tag>> {
  using ConeT = LogDetCone<Space, Tag>;
  using T = linalg::space_type_t<Space>;

  static constexpr ConeCheck is_closed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() ? ConeCheck::True : ConeCheck::False;
  }

  /**
   * @brief pointed test
   *
   * K = { X \in S^n_++, t \in R | log(det(X)) >= t }
   * -K = { -X \in S^n_--, -t \in R | log(det(X)) <= t }
   *
   * K \cap -K = { 0 }
   */
  static constexpr ConeCheck is_pointed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }
};

/// @brief power cone
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<PowerCone<Space, Tag>> {
  using ConeT = PowerCone<Space, Tag>;
  using T = linalg::space_type_t<Space>;

  static constexpr ConeCheck is_closed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() ? ConeCheck::True : ConeCheck::False;
  }

  /**
   * @brief pointed test
   *
   * K = { (x, t) \in R^n_+ x R | |t| <= \prod_i^n x_i^\alpha_i, \sum_i^n
   * \alpha_i <= 1,
   * \forall \alpha >= 0 }
   * -K = { (-x, -t) \in R^n_+ x R | |-t| <= \prod_i^n -x_i^\alpha_i, \sum_i^n
   * \alpha_i <= 1,
   * \forall \alpha >= 0 }
   *
   * -K = { 0 } so K \cap -K = { 0 }
   */
  static constexpr ConeCheck is_pointed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }
};

/// @brief log sum exp cone
template <linalg::RnSpace Space, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<LogSumExpCone<Space, Tag>> {
  using ConeT = LogSumExpCone<Space, Tag>;
  using T = linalg::space_type_t<Space>;

  static constexpr ConeCheck is_closed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() ? ConeCheck::True : ConeCheck::False;
  }

  static constexpr ConeCheck is_pointed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }
};

/// @brief affine subspace -> zero cone
template <linalg::RnSpace Space, linalg::MatrixLike Mat, ViewTag Tag>
  requires(Tag != ViewTag::ConvexSet)
struct cone_traits<AffineSubspace<Space, Mat, Tag>> {
  using ConeT = AffineSubspace<Space, Mat, Tag>;

  static constexpr ConeCheck is_closed(const ConeT& /*cone*/) {
    return ConeCheck::True;
  }

  static constexpr ConeCheck is_solid(const ConeT& cone) {
    return cone.is_well_formed() ? ConeCheck::True : ConeCheck::False;
  }

  /**
   * @brief pointed test
   *
   * K \cap -K = { 0 }
   */
  static constexpr ConeCheck is_pointed(const ConeT& /* cone */) {
    return ConeCheck::True;
  }
};

}  // namespace convex
