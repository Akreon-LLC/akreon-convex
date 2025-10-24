/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/**
 * @brief Ellipsoid defined as the following
 *
 * E = { x | (x - c)^T P (x - c) <= 1 } s.t. P \in S^n_+ (symmetric positive
 * definite)
 */
template <linalg::RnSpace Space, EllipsoidType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Ellipsoid : public ConvexSet<Space, Ellipsoid<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Ellipsoid<Space, Type, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;
  using ConvexBase::N;

  /// @brief common mat type for P
  using Mat = linalg::Matrix<T, N, N>;

  /**
   * @brief CTOR for ellipsoidal set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param P \in S^n_+
   */
  constexpr explicit Ellipsoid(const Point& c_in, const Mat& P_in)
      : c(c_in), P(P_in) {}

  /**
   * @brief contains function to see if x is in the ellipsoid (x - c)^T P^{-1}
   * (x - c) <= 1
   *
   * @param point x of type T with n elements
   * @param tolerance of tyep T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /*tol*/) const {
    const auto delta = as_col_vector(x - c);
    const auto delta_t = linalg::transpose(delta);

    const auto val = (delta_t * (linalg::inverse(P) * delta));
    static_assert(linalg::mat_rows_v<decltype(val)> == 1 &&
                  linalg::mat_cols_v<decltype(val)> == 1);

    switch (Type) {
      case EllipsoidType::OE:
        return val(0)(0) < 1;
      case EllipsoidType::CE:
      default:
        return val(0)(0) <= 1;
    }
    return false;
  }

  /**
   * @brief contains function to see if x is in the ellipsoid (x - c t)^T P^{-1}
   * (x - c t) <= t^2
   *
   * @param (x, t) \in R^n x R
   * @param tolerance of tyep T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x = std::get<0>(x_in);
    auto t = std::get<1>(x_in);
    if (t < T(0)) return false;

    const auto delta = as_col_vector(x - (t * c));
    const auto delta_t = linalg::transpose(delta);

    const auto val = (delta_t * (linalg::inverse(P) * delta));
    static_assert(linalg::mat_rows_v<decltype(val)> == 1 &&
                  linalg::mat_cols_v<decltype(val)> == 1);

    // (dakre) (x - ct)^T P^{-1} (x - ct) <= t^2
    switch (Type) {
      case EllipsoidType::OE:
        return val(0)(0) < t * t;
      case EllipsoidType::CE:
      default:
        return val(0)(0) <= t * t;
    }
    return false;
  }

  /**
   * @brief is_well_formed implementation checks to see if set is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    linalg::SymmetricMatrix S(P);
    return S.type == linalg::SymmetricMatrixType::PD ||
           S.type == linalg::SymmetricMatrixType::PSD;
  }

  /**
   * @brief transform set from an affine map
   *
   * E = {x | (x - c)^T P (x - c) <= 1}
   * f(x) = U x + t -> { y = f(x) | x \in E }
   * x = U^{-1}(y - t), sub back in we get...
   * E' = {y | (U^{-1}(y - t) - c)^T P (U^{-1}(y - t) - c) <= 1}
   * ... U^{-1}(y - t) - c \to U^{-1}(y - (U c + t)
   * E' = { y | (y - (U c + t))^T U^{-T} P U^{-1} (y - (U c + t)) <= 1
   *
   * c' = U c + t
   * P' = U^{-T} P u
   *
   * @param Affine map
   *
   * @return transformed ellipsoid
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<T, linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto U_inv = linalg::inverse(map.linear_part());
    return Ellipsoid<Space, Type>(map.linear_part() * c + map.translation(),
                                  transpose(U_inv) * P * U_inv);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().c == c && other.self().P == P;
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    return ConeView(Ellipsoid<Space, Type, ViewTag::ConvexCone>(c, P));
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
  project_impl(const ProjectiveMap<MatA, MatC>& map, const Point& x) const {
    if (!this->contains(x)) return std::nullopt;
    return map.apply(x);
  }

 private:
  /** @brief member variables in the ellipsoid */
  Point c;  //< n-dim vector of type T representing center of ellipsoid
  Mat P;    //< square matrix representing directions for
            // ellipsoid (valid \in S^n_+)
};

}  // namespace convex
