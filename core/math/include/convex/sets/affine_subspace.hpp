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
    const auto A_tilde = A * linalg::Inverse<TMat>()(map.linear_part());
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
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::SET>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;

    static constexpr size_t ChildCodom =
        linalg::mat_cols_v<decltype(child_coord.linear_part())>;
    static_assert(ChildCodom == N,
                  "Convex set domain does not equal the child coordinate "
                  "linear map's codomain in size");

    linalg::Matrix<T, 2 * M, N> F;
    linalg::Vector<T, 2 * M> g;
    const auto composed_map = affine::AffineMap<Mat>(A, b).compose(child_coord);
    const auto F_eq = composed_map.linear_part();
    const auto g_eq = T(-1) * composed_map.translation();
    F = linalg::set_block<0, M, 0, N>(F, T(-1) * F_eq);
    F = linalg::set_block<M, 2 * M, 0, N>(F, F_eq);
    for (size_t i = 0; i < M; ++i) {
      g(i) = T(-1) * g_eq(i);
      g(M + i) = g_eq(i);
    }

    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 2 * M>>();

    const auto new_builder =
        conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
                                decltype(map), decltype(child_cb)>(map,
                                                                   child_cb);
    const auto cones = std::make_tuple(rplus);
    return new_builder.template make<Type>(cones);
  }

  /** @brief data members for affine subspace */
  Mat A;         //< basis that spans the space of R^n
  AmbientVec b;  //< offset vector which translates the basis to a new origin
};

}  // namespace convex
