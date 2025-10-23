/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../../linalg/symmetric_matrix.hpp"
#include "convex_set.hpp"

namespace convex {

/**
 * @brief Log(det(x)) cone is the hypograph representation of the log det
 * concave function
 *
 * hypo(F(x)) = { (X, t) | log(det(X)) >= t, X \in S^n_++ }
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::Hypograph>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Space>> &&
           Tag == ViewTag::Hypograph)
class LogDetCone : public ConvexSet<Space, LogDetCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, LogDetCone<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

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
  constexpr bool contains_impl(const std::tuple<Point, T>& lifted_X,
                               const T& /* tol */) const {
    const auto X = std::get<0>(lifted_X);
    linalg::SymmetricMatrix S(X);
    if (S.type != linalg::SymmetricMatrixType::PD) return false;
    if (!S.det.has_value()) return false;
    const auto t = std::get<1>(lifted_X);
    return linalg::constexpr_ln(S.det.value()) >= t;
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& /* other */) const {
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
    requires(Tag == ViewTag::ConvexSet)
  constexpr auto transform_impl(const AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    return ConeView(LogDetCone<Space, Tag>());
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
  project_impl(const ProjectiveMap<MatA, MatC>& /*map*/,
               const Point& /*x*/) const {
    return std::nullopt;
  }
};
}  // namespace convex
