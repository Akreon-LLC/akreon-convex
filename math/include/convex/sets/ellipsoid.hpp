/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/// @brief simple enum to describe type of ellipsoid
enum struct EllipsoidType { CE, OE };

/**
 * @brief Ellipsoid defined as the following
 *
 * E = { x | (x - c)^T P (x - c) <= 1 } s.t. P \in S^n_+ (symmetric positive
 * definite)
 */
template <linalg::VectorSpace Space, EllipsoidType Type,
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

  /**
   * @brief CTOR for ellipsoidal set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param P \in S^n_+
   */
  constexpr explicit Ellipsoid(const Point& c_in,
                               const linalg::Matrix<T, N, N>& P_in)
    requires(Tag == ViewTag::ConvexSet)
      : c(c_in), P(P_in) {}

  /**
   * @brief CTOR for ellipsoidal set in conic view
   *
   * @param Point (center)
   * @param P \in S^n_+
   * @param t scalar for cone
   */
  constexpr explicit Ellipsoid(const Point& c_in,
                               const linalg::Matrix<T, N, N>& P_in,
                               const T& t_in)
    requires(Tag == ViewTag::ConvexCone)
      : c(c_in), P(P_in), t(t_in) {}

  /**
   * @brief contains function to see if x is in the ellipsoid (x - c)^T P (x -
   * c) <= 1

   * @param point x of type T with n elements
   * @param tolerance of tyep T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /*tol*/) const {
    T alpha(1);
    if (t.has_value()) alpha *= t.value();

    const auto delta = as_col_vector(x - (alpha * c));
    const auto delta_t = linalg::transpose(delta);

    const auto val = (delta_t * (linalg::inverse(P) * delta));
    static_assert(linalg::mat_rows_v<decltype(val)> == 1 &&
                  linalg::mat_cols_v<decltype(val)> == 1);
    switch (Type) {
      case EllipsoidType::OE:
        return val(0)(0) < alpha;
      case EllipsoidType::CE:
      default:
        return val(0)(0) <= alpha;
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
    if (t.has_value())
      if (t <= T(0)) return false;

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
             std::same_as<T, linalg::mat_type_t<Mat>> &&
             Tag == ViewTag::ConvexSet)
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
    return other.self().c == c && other.self().P == P && other.self().t == t;
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * @param t cone slice
   *
   * @return cone
   */
  constexpr auto cone_view_impl(const T& t_in) const {
    return ConeView(Ellipsoid<Space, Type, ViewTag::ConvexCone>(c, P, t_in));
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
  linalg::Matrix<T, N, N> P;  //< square matrix representing directions for
                              // ellipsoid (valid \in S^n_+)
  std::optional<T> t;         //< conic view t scalar
};

}  // namespace convex
