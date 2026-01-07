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
using AffineDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on affine subspace
template <linalg::RnSpace Space>
using AffineCodom = ConvexDomain<Space, SignType::FREE>;

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
    const auto rank = linalg::QR(A).get_rank();
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
    const auto A_tilde = A * linalg::inverse(map.linear_part());
    const auto b_tilde = b + A_tilde * map.translation();
    return AffineSubspace<Meta, Child>(A_tilde, b_tilde,
                                       std::get<0>(this->children));
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * [ A -b ] [ x; t ] = 0
   * [ 0  1 ]
   *
   * [A x - b t; t], where Ax - bt = 0, t >= 0
   *
   * @return cone block
   */
  template <OrigType Type, BuilderLike Builder>
  constexpr auto canonicalize_set(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<OrigType::SET>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;

    static constexpr size_t ChildCodom =
        linalg::mat_cols_v<decltype(child_coord.linear_part())>;
    static_assert(ChildCodom == N,
                  "Convex set domain does not equal the child coordinate "
                  "linear map's codomain in size");

    linalg::Matrix<T, M + 1, N + 1> F;
    linalg::Vector<T, M + 1> g;
    const auto composed_map = affine::AffineMap<Mat>(A, b).compose(child_coord);
    F = linalg::set_block<0, M, 0, N>(F, composed_map.linear_part());
    F = linalg::set_block<0, M, N, N + 1>(
        F, linalg::as_col_vector(T(-1) * composed_map.translation()));
    F[M][N] = T(1);

    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto zoc = conic::ZeroOrderCone<linalg::Vector<T, M>>();
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();

    const auto new_builder =
        ConeBlockBuilder<builder_dom_t<decltype(builder)>, decltype(map),
                         decltype(child_cb)>(map, child_cb);
    const auto new_map = new_builder.template update_map<Type>();
    return conic::ConeBlock(new_map, std::make_tuple(zoc, rplus));
  }

  /** @brief data members for affine subspace */
  Mat A;         //< basis that spans the space of R^n
  AmbientVec b;  //< offset vector which translates the basis to a new origin
};

}  // namespace convex
