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
#include "linalg/symmetric_matrix.hpp"

namespace convex {

/**
 * @brief default coord map for log(det) cone
 *
 * coords: (x, t) -> log(det(x)) <= t, x \in S^n_+
 */
template <linalg::SquareMatrixLike CoordMat>
constexpr auto default_log_det_map(const linalg::mat_type_t<CoordMat>& t) {
  using T = linalg::mat_type_t<CoordMat>;
  static constexpr size_t N = linalg::mat_rows_v<CoordMat>;
  auto I = linalg::make_identity<T, N>();
  I[N - 1][N - 1] = t;
  return affine::AffineMap<CoordMat>(I, linalg::Vector<T, N>{});
}

/// @brief Log Det Functor
template <linalg::RnSpace Domain>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Domain>>)
struct LogDet {
  using T = linalg::space_type_t<Domain>;
  using Point = linalg::space_point_t<Domain>;

  /**
   * @brief compute the log det of a symmetric matrix
   *
   * @param X \in S^n_++
   *
   * @return optional scalar \in R
   */
  constexpr std::optional<T> operator()(const Point& X) const {
    linalg::SymmetricMatrix S(X);
    if (S.type != linalg::SymmetricMatrixType::PD) return std::nullopt;
    if (!S.det.has_value()) return std::nullopt;
    if (!linalg::is_real(S.det.value())) return std::nullopt;
    return linalg::constexpr_ln(S.det.value().real());
  }
};

/**
 * @brief Log(det(x)) cone
 *
 * hypo(F(x)) = { (X, t) | log(det(X)) >= t, X \in S^n_++ }
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::LiftedCone>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Space>>)
class LogDetCone : public ConvexSet<Space, LogDetCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, LogDetCone<Space, Tag>>;
  using typename ConvexBase::CoordMat;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /**
   * @brief CTOR for log det cone
   */
  constexpr explicit LogDetCone()
      : ConvexBase(default_log_det_map<CoordMat>(T(1))) {}

  /**
   * @brief CTOR for log det cone
   *
   * @param affine map in
   */
  constexpr explicit LogDetCone(const affine::AffineMap<CoordMat>& map_in)
      : ConvexBase(map_in) {}

  /**
   * @brief is_well_formed implementation stubbed to true
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief check if candidate matrix X \in S^n_++
   *
   * @param point in vector space S^n
   * @param scalar tol
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& X, const T& /* tol */) const {
    return contains_impl(std::make_tuple(X, T(1)));
  }

  /**
   * @brief check if candidate matrix X \in S^n_++
   *
   * Standard cone:
   * C' = { (x, t) \in S^n_++ x R | log(det(X)) >= t }
   *
   * @param (x,t) \in S^n_++ x R
   * @param scalar tol
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /* tol */) const {
    const auto x_proj = this->project(x_in);
    const auto x = std::get<0>(x_proj);
    const auto s = std::get<1>(x_proj);

    const auto res = LogDet<Space>()(x);
    if (!res.has_value()) return false;

    return res.value() >= s;
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.coord_map == this->coord_map;
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
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(LogDetCone<Space, ViewTag::LiftedCone>(this->coord_map));
    else
      return ConeView(LogDetCone<Space, Tag>(this->coord_map));
  }
};

}  // namespace convex
