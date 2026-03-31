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
template <typename Meta, typename Child, linalg::SquareMatrixLike MatIn>
struct LMI : ConvexSet<Meta, LMI<Meta, Child, MatIn>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, LMI<Meta, Child, MatIn>, Child>;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::N;

  /// @brief lifted matrix for P
  using Mat = linalg::remove_cvref_t<MatIn>;
  static constexpr size_t MatDim = linalg::mat_rows_v<Mat>;
  static_assert(linalg::mat_cols_v<Mat> == MatDim,
                "LMI expects square coefficient matrices");

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

    static constexpr size_t K = MatDim * (MatDim + 1) / 2;
    linalg::Matrix<T, K, N> F{};
    linalg::Vector<T, K> g{};

    auto update_F = [&F](const auto& i, const auto&& ldl) {
      size_t row_idx{0};
      for (size_t j = 0; j < MatDim; ++j) {
        for (size_t k = j; k < MatDim; ++k, ++row_idx) {
          if (k == j)
            F(row_idx)(i) = T(-1) * ldl(k, j);
          else
            F(row_idx)(i) =
                T(-1) * linalg::constexpr_sqrt_safe(T(2)) * ldl(k, j);
        }
      }
    };

    auto update_g = [&g](const auto& ldl) {
      size_t row_idx{0};
      for (size_t j = 0; j < MatDim; ++j) {
        for (size_t k = j; k < MatDim; ++k, ++row_idx) {
          if (k == j)
            g(row_idx) = ldl(k, j);
          else
            g(row_idx) = linalg::constexpr_sqrt_safe(T(2)) * ldl(k, j);
        }
      }
    };

    for (size_t i = 0; i < N; ++i) update_F(i, std::move(linalg::LDL(A(i))));
    update_g(std::move(linalg::LDL(B)));

    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto psd_cone =
        conic::PositiveSemidefiniteCone<linalg::Vector<T, K>, MatDim>();
    const auto cones = std::make_tuple(psd_cone);

    const auto new_builder =
        conic::CanonicalFormBuilder<conic::builder_dom_t<decltype(builder)>,
                                    decltype(map), decltype(child_cb),
                                    decltype(child_builder.eq_map)>(
            map, child_cb, child_builder.eq_map);
    return new_builder.template make<Type>(cones);
  }

  /** @brief member vars for LMI */
  linalg::Vector<Mat, N> A;  ///< LMI of As
  Mat B;                     ///< B offset
};

/// (dakre) CTAD - class template argument deduction for egononic use of LMI
/// instead of full template path
template <typename Mat, size_t N, typename Child>
LMI(const linalg::Vector<Mat, N>&, const Mat&, const Child&)
    -> LMI<LMIMeta<dom_space_t<meta_codom_t<atom_meta_t<Child>>>>, Child, Mat>;

}  // namespace convex
