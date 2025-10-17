/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/**
 * @brief Hyperplane defition defined on valid vector space
 *
 * H = { x | a^T x = b }
 */
template <linalg::VectorSpace Space, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Hyperplane : public ConvexSet<Space, Hyperplane<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Hyperplane<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief hyperplane dimension
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /// @brief lifted point (based on lifted space or not)
  using AugmentedPoint = std::conditional_t<Tag == ViewTag::ConvexCone,
                                            linalg::Vector<T, N - 1>, Point>;

  /**
   * @brief CTOR for hyperplane
   *
   * @param Point (i.e., n-dim vec of type T)
   * @param b bias scalar
   */
  constexpr explicit Hyperplane(const AugmentedPoint& a_in, const T& b_in)
      : a(a_in), b(b_in) {}

  /**
   * @brief contains function to see if x lies in the hyerplane defined by a^T x
   * == b
   *
   * When lifted to convex cone the pair (x,t) \in R^{n + 1}, so last elem of
   * point x is t
   *
   * @param point x of type T (n-dim)
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x_in, const T& tol) const {
    T t(b);
    AugmentedPoint x;
    if constexpr (Tag == ViewTag::ConvexCone) {
      if (x_in(N - 1) < T(0)) return false;
      t *= x_in(N - 1);
      x = std::get<0>(linalg::splice<N - 1>(x_in));
    } else {
      x = x_in;
    }

    return linalg::constexpr_abs(linalg::dot(a, x) - t) < tol;
  }

  /**
   * @brief contains dual impl
   *
   * K = { (x,t) | a^T x = b t } \to { z | M z = 0 } where M = [ a^T -b ]
   * Linear subspace L = {z | M z = 0} the dual is the orthogonal complement
   * (subspaces annihilator) which comprises of linear functionals that map
   * every vector in the subspace to the nullspace { 0 }
   *
   * L = { z | M z = 0}, L^* = range(M^T)
   * y \in K^* iff x^T y = 0
   */
  constexpr bool contains_dual_impl(const Point& y, const T& tol) const
    requires(Tag == ViewTag::ConvexCone)
  {
    Point m;
    m[N - 1] = T(-1) * y(N - 1) * b;
    for (size_t i = 0; i < N - 1; ++i) m[i] = a(i);
    const auto mTm = linalg::dot(m, m);
    if (mTm == 0) return false;
    const auto y_proj = (dot(y, m) / mTm) * m;
    return L2<T, N - 1>(y_proj) <= tol;
  }

  /**
   * @brief is well formed checks to see if the hyperplane is valid
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    if (linalg::all_zero(a)) return false;
    return true;
  }

  /**
   * @brief transform implementation
   *
   * H = { x | a^T x = b }
   * f(H) = { y = f(x) | x \in H }
   * y = U x + t -> x = U^{-1} (y - t)
   * a^T U^{-1} (y - t) = b \to a^T U^{-1} y = b + a^T U^{-1}
   *
   * @param affine map
   *
   * @return transformed hyperplane
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<linalg::mat_type_t<Mat>, T> &&
             Tag == ViewTag::ConvexSet)
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Hyperplane<Space>(as_vector(aT_U),
                             b + (aT_U * map.translation())(0));
  }

  /**
   * @brief is_equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().a == a && other.self().b == b;
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const
    requires(Tag == ViewTag::ConvexSet)
  {
    return ConeView(
        Hyperplane<linalg::Matrix<T, 1, N + 1>, ViewTag::ConvexCone>(a, b));
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
    requires(Tag == ViewTag::ConvexSet)
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  project_impl(const ProjectiveMap<MatA, MatC>& map, const Point& x) const {
    if (!this->contains(x)) return std::nullopt;
    return map.apply(x);
  }

 private:
  /** @brief member var in hyperplane */
  AugmentedPoint a;  //< n-dim vec of type T
  T b;               //< bias scalar of type T
};

}  // namespace convex
