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
using LMIDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on affine subspace
template <linalg::RnSpace Space>
using LMICodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for meta on affine subspace
template <linalg::RnSpace Space>
using LMIMeta =
    ConvexAtomMeta<LMIDom<Space>, LMICodom<Space>, CurvatureType::CONVEX>;

/**
 * @brief LMI (linear matrix inequality)
 *
 *
 * LMI solution set: 0 \succeq B - \sum(x_i A_i)
 */
template <typename Meta, typename Child>
struct LMI : ConvexSet<Meta, LMI<Meta, Child>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, LMI<Meta, Child>, Child>;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::N;

  /// @brief lifted matrix for P
  using Mat = linalg::Matrix<T, N, N>;

  /**
   * @brief CTOR for LMI A & B split out
   *
   * @param A vector of matrices
   * @param B matrix
   */
  constexpr explicit LMI(const linalg::Vector<Mat, N>& A_in, const Mat& B_in,
                         const Child& child_in)
      : ConvexBase(child_in), A(A_in), B(B_in) {}

  /**
   * @brief contains function to see if x is in the LMI defined as
   *
   * LMI
   * (i)  PSD: 0 <= B - A(x)
   * (ii) PD: 0 < B - A(x)
   *
   * @param x matrix with N vectors
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& /* tol */) const {
    auto res = B;
    for (size_t i = 0; i < N; ++i) res = res - (x(i) * A(i));
    linalg::LDL ldl(res);
    return ldl.is_psd();
  }

  /**
   * @brief is_well_formed to check if the set is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const {
    if (!linalg::LDL(B).is_valid()) return false;
    for (size_t i = 0; i < N; ++i)
      if (!linalg::LDL(A(i)).is_valid()) return false;
    return true;
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    if (!linalg::all_equal(B, other.B)) return false;
    for (size_t i = 0; i < N; ++i)
      if (!linalg::all_equal(A(i), other.A(i))) return false;
    return true;
  }

  /**
   * @brief transform impl
   *
   * @param map
   *
   * @return new set
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    AK_ASSERT(false && "LMI does not support transformations");
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * [old] deprecating homogenized t
   * C = { X in S^n_+ | A(x) <= B }
   * K = { (X, t) | A(x) <= B t }... linear operator-> { (x, t) | B t - sum(x_i
   * A_i) >= 0, t >= 0 }
   *
   * a_i = svec(A_i) in R^k A_i in S^n where k = n(n+1)/2
   * b = svec(B) ...
   *
   * B t - sum(x_i A_i) >= 0 ... psd cone
   * t >= 0... nonneg orthant cone
   *
   * [ -a_i ... -a_n b ] [ x; t ] -> (u, t)
   * [ 0    ...    0 1 ]
   *
   * smat(u) >= 0, t>= 0
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize(const Builder& builder) const {
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

    static constexpr size_t K = N * (N + 1) / 2;
    linalg::Matrix<T, K, N> F;
    for (size_t i = 0; i < N; ++i) {
      linalg::LDL ldl(A(i));
      for (size_t j = 0; j < K; ++j) F(j)(i) = T(-1) * ldl.svec(j);
    }
    linalg::LDL ldl(B);
    linalg::Vector<T, K> g{};
    for (size_t i = 0; i < K; ++i) g(i) = ldl.svec(i);
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto psd_cone =
        conic::PositiveSemidefiniteCone<linalg::Vector<T, K>, N>();
    const auto cones = std::make_tuple(psd_cone);

    const auto new_builder =
        conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
                                decltype(map), decltype(child_cb)>(map,
                                                                   child_cb);
    return new_builder.template make<Type>(cones);
  }

  /** @brief member vars for LMI */
  linalg::Vector<Mat, N> A;  ///< LMI of As
  Mat B;                     ///< B offset
};

}  // namespace convex
