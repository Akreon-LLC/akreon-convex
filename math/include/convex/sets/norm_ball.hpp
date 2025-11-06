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
#include "ellipsoid.hpp"
#include "norm_cone.hpp"

namespace convex {

/**
 * @brief Norm ball defined on valid vector space and one of the following type
 * traits
 *
 * { x | norm(x - c) <= r }... closed norm (L2, L1, Linf) ball that lies within
 * radius { x | norm(x - c) < r}... open "" ball that ""
 */
template <linalg::RnSpace Space, NormBallType Type,
          ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           Tag == ViewTag::ConvexSet)
class NormBall : public ConvexSet<Space, NormBall<Space, Type, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NormBall<Space, Type, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for NormBall set
   *
   * @param Point (i.e., n-dim vector of type T)
   */
  constexpr explicit NormBall(const Point& c_in, const T& r_in)
      : c(c_in), r(r_in) {}

  /**
   * @brief contains function to see if x is in the norm ball (norm(x - c) <= r)
   *
   * @param Point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /* tol */) const {
    switch (Type) {
      case NormBallType::L2OB:
        return linalg::L2<Point>()(x - c) < r;
      case NormBallType::L2CB:
        return linalg::L2<Point>()(x - c) <= r;
      case NormBallType::L1OB:
        return linalg::L1<Point>()(x - c) < r;
      case NormBallType::L1CB:
        return linalg::L1<Point>()(x - c) <= r;
      case NormBallType::LinfOB:
        return linalg::Linf<Point>()(x - c) < r;
      case NormBallType::LinfCB:
      default:
        return linalg::Linf<Point>()(x - c) <= r;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex set
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return r >= 0; }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().c == c && other.self().r == r;
  }

  /**
   * @brief transform norm ball by affine map
   *
   * Br(c) = ||x - c|| < r or B = {x | ||x - c|| < r}
   * f(B) = {y = f(x) | x \in B}... y = Ux + t -> U^{-1}(y - t) = x
   * Br'(c) = ||U^{-1}(y - t) - c|| < r -> U^{-1}(y - (U c + t))

   * Above is a polyhedron represented in Ax <= b and Cx = d by projecting onto
   * y-coordinates
   *
   * @param affine map
   *
   * @return transformed ball (ellipse for L2, polyhedron for L1/Linf)
   */
  template <linalg::SquareMatrixLike Mat>
    requires(linalg::mat_cols_v<Mat> == N &&
             std::same_as<T, linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto& U = map.linear_part();
    const auto& t = map.translation();
    /* L2 case
     * Let z = y - (U c + t) \implies ||U^{-1} z||_2^2 <= r^2 \iff z^T (U^-T
     * U^-1) z <= r^2 So we have f(Br(x)) = {y | (y - (U c + t))^T (U U^T)^{-1}
     * (y - (U c + t))  <= r^2
     *
     * (i)  c' = U c + t
     * (ii) r' = ||U^{-1}|| \to r \sigma(U) OR U U^T r^2
     */
    if constexpr (Type == NormBallType::L2OB || Type == NormBallType::L2CB) {
      const auto point = U * c + t;
      Mat U_T;
      if constexpr (linalg::is_complex_v<T>)
        U_T = conjugate_transpose(U);
      else
        U_T = transpose(U);
      const auto P = (r * r) * (U * U_T);

      if constexpr (Type == NormBallType::L2OB)
        return Ellipsoid<Space, EllipsoidType::OE>(point, P);
      else
        return Ellipsoid<Space, EllipsoidType::CE>(point, P);
    }
    /* L1 case
     *
     * L1 ball now represented as a polyhedron (LP equalities & inequalities):
     * We have || U^{-1} z || <= r from above, where c transforms similarly to
     * the above however r scales differently now (forms a type of polyhedron).
     *
     * Note the following relationships for L1:
     * || U^{-1} z ||_1 \to \sum_{i=1}^n | U^{-1} z | <= r
     * let s = [s_0 s_1 ... s_n] >= 0 not use slack var s to represent abs value
     * respecting the constraint \sum_i^n s_i <= r
     * furthermore note z = y - c', c' = Uc + t so the following constraints
     * are:
     *
     * U^{-1} z \to U^{-1} y - U^{-1} U c - t
     *
     * (i)   1^T s <= r
     * (ii)  -(U^{-1} z)_i <= s_i \to -U^{-1} y - s <= -(c + U^{-1} t)
     * (iii) (U^{-1} z)_i <= s_i \to U^{-1} y - s <= (c + U^{-1} t)
     * (iv) s >= 0 \to -s <= 0
     *
     * P = { x | Ax <= b, Cx = d } where x has an expanded dim including s
     * x = [y s] \in R^{2*n} and b = [r 0 0 0] \in R^{3*n + 1}
     *
     * Affine Inequalities:
     * (i)   A_1 = [ 0^T; 1^T ], b_1 = r ... first row in A
     * (ii)  A_2 = [ -U^{-1}; -I ], b_2 = -(c + U^{-1} t)
     * (iii) A_3 = [ U^{-1}; -I ], b_3 = (c + U^{-1} t)
     * (iv)  A_4 = [ 0; -I ], b_4 = 0
     *
     * Linear Equality C x = d, where z = y - c'... c' = U x + t
     *
     * L1 in Complex domain is different where |z_i| = sqrt(re(z)^2 + im(z)^2)
     * so... \sum_i |z_i| <= r transforms to Second order cone program:
     *
     * z = y - c'... c' = U c + t
     *
     * (i) ||z_i||_2 <= s_i
     * (ii) \sum_i s_i <= r
     */
    else if constexpr (Type == NormBallType::L1OB ||
                       Type == NormBallType::L1CB) {
      linalg::Matrix<T, 3 * N + 1, 2 * N> A;
      linalg::Vector<T, 3 * N + 1> b;
      b[0] = r;

      const auto U_inv = linalg::inverse(U);
      const auto c_plus_U_inv_t = c + U_inv * t;
      for (size_t i = 1; i < 2 * N + 1; ++i) {
        const size_t idx = (i - 1) % N;
        if (i < N + 1)
          b[i] = T(-1) * c_plus_U_inv_t(idx);
        else
          b[i] = c_plus_U_inv_t(idx);
      }

      const auto I = linalg::make_identity<T, N>();
      // (i) first row assignment 1^T s <= r
      for (size_t i = 0; i < N; ++i) A[0][N + i] = T(1);
      // (ii) first block assignment -U^{-1} z - s <= 0
      A = linalg::set_block<1, N + 1, 0, N>(A, T(-1) * U_inv);
      A = linalg::set_block<1, N + 1, N, 2 * N>(A, T(-1) * I);
      // (iii) second block assignment U^{-1} z - s <= 0
      A = linalg::set_block<N + 1, 2 * N + 1, 0, N>(A, U_inv);
      A = linalg::set_block<N + 1, 2 * N + 1, N, 2 * N>(A, T(-1) * I);
      // (iv) third block assignment s >= 0
      A = linalg::set_block<2 * N + 1, 3 * N + 1, N, 2 * N>(A, T(-1) * I);

      using NewSpace = linalg::Matrix<T, 1, 2 * N>;
      if constexpr (Type == NormBallType::L1OB)
        return Polyhedron<NewSpace, HalfspaceType::OLH, decltype(A),
                          linalg::Matrix<T, 0, 2 * N>, Tag>(A, b);
      else
        return Polyhedron<NewSpace, HalfspaceType::CLH, decltype(A),
                          linalg::Matrix<T, 0, 2 * N>, Tag>(A, b);
    }
    /* Linf case
     *
     * Real case starts similar to L1 but no need for a vector of slack vars
     * as Linf encodes maximum abs value for the full matrix.
     *
     * ||U^{-1} z||_inf = max |U^{-1} z|... largest magnitude among all
     * components
     *
     * z = y - c'... c' = Uc + t so, U^{-1} y - U^{-1}(U c + t)
     *
     * -U^{-1} z <= r \to -U^{-1} y <= r - U^{-1}(U c + t)
     *  U^{-1} z <= r \to U^{-1} y <= r + U^{-1}(U c + t)
     *
     * Affine Inequality is Ax <= b, x = [y]
     * A = [ U^{-1} ; -U^{-1} ]
     * b = [ r + c + U^{-1} t; r - c - U^{-1} t ]
     *
     * Complex case |z| <= t -> ||z||_2 <= t
     * (U^{-1} (y - (U c + t)))^T (U^{-1} (y - (U c + t))) <= t^2
     *
     */
    else {
      linalg::Matrix<T, 2 * N, N> A;
      linalg::Vector<T, 2 * N> b;

      const auto U_inv = linalg::inverse(U);
      const auto c_plus_U_inv_t = c + U_inv * t;
      for (size_t i = 0; i < 2 * N; ++i) {
        const auto idx = i % N;
        if (i < N)
          b[i] = r + c_plus_U_inv_t(idx);
        else
          b[i] = r - c_plus_U_inv_t(idx);
      }

      A = linalg::set_block<0, N, 0, N>(A, U_inv);
      A = linalg::set_block<N, 2 * N, 0, N>(A, T(-1) * U_inv);

      using NewSpace = linalg::Matrix<T, 1, N>;
      if constexpr (Type == NormBallType::LinfOB)
        return Polyhedron<NewSpace, HalfspaceType::OLH, decltype(A),
                          linalg::Matrix<T, 0, N>, Tag>(A, b);
      else
        return Polyhedron<NewSpace, HalfspaceType::CLH, decltype(A),
                          linalg::Matrix<T, 0, N>, Tag>(A, b);
    }
  }

  /**
   * @brief inverse perspective func from convex set to cone
   *
   * @param t cone slice
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Type == NormBallType::L2OB)
      return ConeView(
          NormCone<Space, NormConeType::L2OC, ViewTag::LiftedCone>(c, r));
    else if constexpr (Type == NormBallType::L2CB)
      return ConeView(
          NormCone<Space, NormConeType::L2CC, ViewTag::LiftedCone>(c, r));
    else if constexpr (Type == NormBallType::L1OB)
      return ConeView(
          NormCone<Space, NormConeType::L1OC, ViewTag::LiftedCone>(c, r));
    else if constexpr (Type == NormBallType::L1CB)
      return ConeView(
          NormCone<Space, NormConeType::L1CC, ViewTag::LiftedCone>(c, r));
    else if constexpr (Type == NormBallType::LinfOB)
      return ConeView(
          NormCone<Space, NormConeType::LinfOC, ViewTag::LiftedCone>(c, r));
    else
      return ConeView(
          NormCone<Space, NormConeType::LinfCC, ViewTag::LiftedCone>(c, r));
  }

 private:
  /** @brief member variables in the norm ball */
  Point c;  //< n-dim vector of type T representing center of ball
  T r;      //< scalar of type T representing radius
};

}  // namespace convex
