/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"
#include "linalg/symmetric_matrix.hpp"

namespace convex {

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
 * @brief Log(det(x)) cone is the hypograph representation of the log det
 * concave function
 *
 * hypo(F(x)) = { (X, t) | log(det(X)) >= t, X \in S^n_++ }
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::LiftedCone>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Space>>)
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
   * (1) Epigraph lift C' = { (x, t) \in S^n_++ x R | log(det(X)) >= -t }
   * (2) Standard & Hypograph lift C' = { (x, t) \in S^n_++ x R | log(det(X)) >=
   * t }
   *
   * @param (x,t) \in S^n_++ x R_++ or S^n_++ x R depending on lift
   * @param scalar tol
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& lifted_X,
                               const T& /* tol */) const {
    const auto X = std::get<0>(lifted_X);
    const auto t = std::get<1>(lifted_X);
    const auto res = LogDet<Space>()(X);
    if (!res.has_value()) return false;

    if constexpr (Tag == ViewTag::EpigraphCone)
      return res.value() <= t;
    else
      return res.value() >= t;
  }

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
      return ConeView(LogDetCone<Space, ViewTag::LiftedCone>());
    else
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
  projective_map_impl(const ProjectiveMap<MatA, MatC>& /*map*/,
                      const Point& /*x*/) const {
    return std::nullopt;
  }
};
}  // namespace convex
