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

/**
 * @brief LMI (linear matrix inequality)
 *
 * A(x) = x_1 A_1 + ... _ x_n A_n <= B
 *
 * A, B \in S^n
 *
 * Solution set of LMI is convex:
 *
 * LMI = { x | A(x) <= B } AKA P(PsdCone) = LMI (P is a valid perspective
 * func)... f(x) = B - A(x)
 *
 * Transformed to pre-image of PsdCone
 * F(x) = F_0 + \sum_i F(x) >= 0... F_0 = B, F_i = -A_i
 */
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class LMI : public ConvexSet<Space, LMI<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, LMI<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;
  using ConvexBase::N;

  /// @brief lifted matrix for P
  using Mat = linalg::Matrix<T, N, N>;

  /**
   * @brief CTOR for LMI
   *
   * @param parameter pack of symmetric matrices in R^n where first param is B
   * and the following are As
   */
  template <typename... Fs>
  constexpr explicit LMI(Fs&&... xs) {
    static_assert(sizeof...(xs) == N + 1);
    int i = 0;
    auto set_mat = [&](const auto& x) {
      if (i == 0)
        B = x;
      else
        A[i - 1] = x;
      i++;
    };
    ((void)set_mat(std::forward<Fs>(xs)), ...);
  }

  /**
   * @brief CTOR for LMI A & B split out
   *
   * @param A vector of matrices
   * @param B matrix
   */
  constexpr explicit LMI(const linalg::Vector<Mat, N>& A_in, const Mat& B_in)
      : A(A_in), B(B_in) {}

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
  constexpr bool contains_impl(const Point& x, const T& /* tol */) const {
    auto res = B;
    for (size_t i = 0; i < N; ++i) res = res - (x(i) * A(i));
    linalg::SymmetricMatrix S(res);
    return S.is_psd();
  }

  /**
   * @brief is_well_formed to check if the set is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    if (linalg::SymmetricMatrix(B).type == linalg::SymmetricMatrixType::INV)
      return false;
    for (size_t i = 0; i < N; ++i)
      if (linalg::SymmetricMatrix(A(i)).type ==
          linalg::SymmetricMatrixType::INV)
        return false;
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
   * @return self
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief helper to get As
   *
   * @return vector of matrices
   */
  constexpr auto get_A() const { return A; }

  /**
   * @brief helper to get B
   *
   * @return B
   */
  constexpr auto get_B() const { return B; }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
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
  constexpr auto canonicalize() const {
    static constexpr size_t K = N * (N + 1) / 2;
    linalg::Matrix<T, K + 1, N + 1> F;
    for (size_t i = 0; i < N; ++i) {
      linalg::SymmetricMatrix S(A(i));
      for (size_t j = 0; j < K; ++j) F[j][i] = T(-1) * S.svec(j);
    }
    linalg::SymmetricMatrix S(B);
    F = linalg::set_block<0, K, N, N + 1>(F, linalg::as_col_vector(S.svec));
    F[K][N] = T(1);
    linalg::Vector<T, K + 1> g;
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto psd_cone =
        conic::PositiveSemidefiniteCone<linalg::Matrix<T, K, N + 1>>();
    const auto rplus = conic::NonnegOrthantCone<linalg::Matrix<T, 1, N + 1>>();
    return std::tuple(conic::ConeBlock(map, psd_cone, rplus));
  }

 private:
  /** @brief member vars for LMI */
  linalg::Vector<Mat, N> A;
  Mat B;
};

}  // namespace convex
