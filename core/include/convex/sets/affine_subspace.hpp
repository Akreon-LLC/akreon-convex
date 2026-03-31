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

/// @brief alias for domain on affine subspace
template <linalg::RnSpace Space>
using AffineDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on affine subspace
template <linalg::RnSpace Space>
using AffineCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for meta on affine subspace
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using AffineMeta =
    ConvexAtomMeta<AffineDom<Dom>, AffineCodom<Codom>, CurvatureType::AFFINE>;

/**
 * @brief Affine Set in the perspective of an affine equality set
 *
 * C = { x | A x = b }
 * C is an affine subspace of R^n where A \in R^{m x n} and b \in R^m
 */
template <typename Meta, typename Child>
struct AffineSubspace : ConvexSet<Meta, AffineSubspace<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexSet<Meta, AffineSubspace<Meta, Child>, Child>;
  using ConvexBase::M;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  static_assert(M == N, "Affine subspace's linear map must be square");

  /// @brief linear map
  using Mat = linalg::Matrix<T, M, N>;

  /// @brief alias for ambient vector
  using AmbientVec = linalg::Vector<T, M>;

  /**
   * @brief CTOR for affine subspace
   *
   * @param A \in R^{m x n}
   * @param b \in R^m
   * @param child
   */
  constexpr explicit AffineSubspace(const Mat& A_in, const AmbientVec& b_in,
                                    const Child& child_in)
      : ConvexBase(child_in), A(A_in), b(b_in) {}

  /**
   * @brief contains function
   *
   * @param point x \in R^n
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& tol) const {
    return linalg::L2<AmbientVec>()(A * x - b) < tol;
  }

  /**
   * @brief is well formed checks to see if basis fully spans subspace
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const {
    const auto rank = linalg::QR(A).rank();
    return rank == N;
  }

  /**
   * @brief equality comparison
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return A == other.self().A && b == other.self().b &&
           other.coord_map == this->coord_map;
  }

  /**
   * @brief transform
   *
   * C = { x | A x = b }
   * f(C) = { y = f(x) | x \in C }
   * y = U x + t \to x = U^{-1} (y - t)
   * A U^{-1} (y - t) = b \to A U^{-1} y = b + A U^{-1} t
   *
   * @param affine map
   *
   * @return transformed affine subspace
   */
  template <linalg::SquareMatrixLike TMat>
    requires(linalg::mat_cols_v<TMat> == N &&
             std::same_as<linalg::mat_type_t<TMat>, T>)
  constexpr auto transform_impl(const affine::AffineMap<TMat>& map) const {
    const auto [U_inv, event] =
        linalg::MatrixInverse<TMat>()(map.linear_part());
    static_cast<void>(event);
    const auto A_tilde = A * U_inv;
    const auto b_tilde = b + A_tilde * map.translation();
    return AffineSubspace<Meta, Child>(A_tilde, b_tilde,
                                       std::get<0>(this->children));
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * [old] deprecating homogeneous t
   * [ A -b ] [ x; t ] = 0
   * [ 0  1 ]
   *
   * [A x - b t; t], where Ax - bt = 0, t >= 0
   *
   * The current PD-IPM path handles equality sets more robustly when they are
   * expressed as paired inequalities:
   *
   *   A x <= b
   *  -A x <= -b
   *
   * @return cone block
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_set(const Builder& builder) const {
    static_cast<void>(Type);
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::SET>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;
    const auto child_eq = child_builder.eq_map;

    static constexpr size_t ChildCodom =
        linalg::mat_cols_v<decltype(child_coord.linear_part())>;
    static_assert(ChildCodom == N,
                  "Convex set domain does not equal the child coordinate "
                  "linear map's codomain in size");

    const auto eq_map = affine::AffineMap<Mat>(A, b).compose(child_coord);

    static constexpr size_t ChildEqRows =
        linalg::mat_rows_v<decltype(child_eq.linear_part())>;
    static constexpr size_t EqRows = ChildEqRows + M;
    static constexpr size_t EqCols =
        linalg::mat_cols_v<decltype(eq_map.linear_part())>;

    linalg::Matrix<T, EqRows, EqCols> A_eq{};
    linalg::Vector<T, EqRows> g_eq{};

    if constexpr (ChildEqRows != 0) {
      A_eq = linalg::set_block<0, ChildEqRows, 0, EqCols>(
          A_eq, child_eq.linear_part());
      for (size_t i = 0; i < ChildEqRows; ++i)
        g_eq(i) = child_eq.translation()(i);
    }

    A_eq = linalg::set_block<ChildEqRows, EqRows, 0, EqCols>(
        A_eq, eq_map.linear_part());
    for (size_t i = 0; i < M; ++i)
      g_eq(i + ChildEqRows) = eq_map.translation()(i);

    const auto new_eq_map = affine::AffineMap<decltype(A_eq)>(A_eq, g_eq);
    return conic::CanonicalFormBuilder<
        conic::builder_dom_t<decltype(builder)>, decltype(child_coord),
        decltype(child_cb), decltype(new_eq_map)>(child_coord, child_cb,
                                                  new_eq_map);
  }

  /** @brief data members for affine subspace */
  Mat A;         //< basis that spans the space of R^n
  AmbientVec b;  //< offset vector which translates the basis to a new origin
};

template <typename Mat, typename Vec, typename Child>
AffineSubspace(const Mat&, const Vec&, const Child&)
    -> AffineSubspace<
        AffineMeta<dom_space_t<meta_codom_t<atom_meta_t<Child>>>,
                   dom_space_t<meta_codom_t<atom_meta_t<Child>>>>,
        const Child>;

}  // namespace convex
