/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/**
 * @brief LMI (linear matrix inequality)
 *
 * A(x) = x_1 A_1 + ... _ x_n A_n <= B
 *
 * A, B \in S^n_+
 *
 * Solution set of LMI is convex:
 *
 * LMI = { x | A(x) <= B } AKA P(PsdCone) = LMI (P is a valid perspective
 * func)... f(x) = B - A(x)
 *
 * Transformed to pre-image of PsdCone
 * F(x) = F_0 + \sum_i F(x) >= 0... F_0 = B, F_i = -A_i
 */
template <linalg::RnSpace Space, LMIType Type, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class LMI : public ConvexSet<Space, LMI<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, LMI<Space, Type, Tag>>;
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
        A[i - 1] = T(-1) * x;
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
  constexpr bool contains_impl(const Point& x, const T& /*tol*/) const {
    auto res = B;
    for (size_t i = 0; i < N; ++i) res = res + (x(i) * A(i));
    linalg::SymmetricMatrix S(res);
    return check_definiteness(S);
  }

  /**
   * @brief contains function to see if x is in the LMI defined as
   *
   * LMI
   * (i)  PSD: 0 <= t * B - A(x)
   * (ii) PD: 0 < t * B - A(x)
   *
   * @param x matrix with N vectors
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x = std::get<0>(x_in);
    const auto t = std::get<1>(x_in);
    auto res = t * B;
    for (size_t i = 0; i < N; ++i) res = res + (x(i) * A(i));
    linalg::SymmetricMatrix S(res);
    return check_definiteness(S);
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
   * @brief project impl
   *
   * @param project map
   * @param point in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  projective_map_impl(const ProjectiveMap<MatA, MatC>& /*map*/,
                      const Point& /*x*/) const {
    return std::nullopt;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    return ConeView(LMI<Space, Type, ViewTag::ConvexCone>(A, B));
  }

  /**
   * @brief helper function to check definiteness of matrix
   *
   * @param symmetric matrix S
   *
   * @return T/F
   */
  constexpr bool check_definiteness(
      const linalg::SymmetricMatrix<Mat>& S) const {
    switch (S.type) {
      case linalg::SymmetricMatrixType::PSD: {
        if (Type == LMIType::NonStrict) return true;
        break;
      }
      case linalg::SymmetricMatrixType::PD:
        return true;
      default:
        break;
    }
    return false;
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

 private:
  /** @brief member vars for LMI */
  linalg::Vector<Mat, N> A;
  Mat B;
};

}  // namespace convex
