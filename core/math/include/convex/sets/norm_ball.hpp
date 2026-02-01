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
#include "polyhedron.hpp"

namespace convex {

/// @brief alias for domain on ellipsoid
template <linalg::RnSpace Space>
using NormBallDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for domain on ellipsoid
template <linalg::RnSpace Space>
using NormBallCodom = NormBallDom<Space>;

/// @brief alias for meta on ellipsoid
template <linalg::RnSpace Space>
using NormBallMeta = ConvexAtomMeta<NormBallDom<Space>, NormBallCodom<Space>,
                                    CurvatureType::CONVEX>;

/**
 * @brief Norm ball defined on valid vector space and one of the following type
 * traits
 *
 * { x | norm(x - c) <= r }... closed norm (L2, L1, Linf) ball that lies within
 * radius { x | norm(x - c) < r}... open "" ball that ""
 */
template <typename Meta, typename Child, NormBallType Type>
struct NormBall : ConvexSet<Meta, NormBall<Meta, Child, Type>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, NormBall<Meta, Child, Type>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR for NormBall set
   *
   * @param Point (i.e., n-dim vector of type T)
   */
  constexpr explicit NormBall(const PointT& c_in, const T& r_in,
                              const Child& child_in)
      : ConvexBase(child_in), c(c_in), r(r_in) {}

  /**
   * @brief contains function to see if x is in the norm ball (norm(x - c) <= r)
   *
   * @param Point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& /* tol */) const {
    switch (Type) {
      case NormBallType::L2:
        return linalg::L2<PointT>()(x - c) <= r;
      case NormBallType::L1:
        return linalg::L1<PointT>()(x - c) <= r;
      case NormBallType::Linf:
      default:
        return linalg::Linf<PointT>()(x - c) <= r;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex set
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const { return r >= 0; }

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
    if constexpr (Type == NormBallType::L2) {
      const auto point = U * c + t;
      Mat U_T;
      if constexpr (linalg::is_complex_v<T>)
        U_T = conjugate_transpose(U);
      else
        U_T = transpose(U);
      const auto P = (r * r) * (U * U_T);

      return Ellipsoid<EllipsoidMeta<linalg::Matrix<T, 1, N>>, Child>(
          point, P, std::get<0>(this->children));
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
    else if constexpr (Type == NormBallType::L1) {
      linalg::Matrix<T, 3 * N + 1, 2 * N> A;
      linalg::Vector<T, 3 * N + 1> b;
      b(0) = r;

      const auto U_inv = linalg::inverse(U);
      const auto c_plus_U_inv_t = c + U_inv * t;
      for (size_t i = 1; i < 2 * N + 1; ++i) {
        const size_t idx = (i - 1) % N;
        if (i < N + 1)
          b(i) = T(-1) * c_plus_U_inv_t(idx);
        else
          b(i) = c_plus_U_inv_t(idx);
      }

      const auto I = linalg::make_identity<T, N>();
      // (i) first row assignment 1^T s <= r
      for (size_t i = 0; i < N; ++i) A(0)(N + i) = T(1);
      // (ii) first block assignment -U^{-1} z - s <= 0
      A = linalg::set_block<1, N + 1, 0, N>(A, T(-1) * U_inv);
      A = linalg::set_block<1, N + 1, N, 2 * N>(A, T(-1) * I);
      // (iii) second block assignment U^{-1} z - s <= 0
      A = linalg::set_block<N + 1, 2 * N + 1, 0, N>(A, U_inv);
      A = linalg::set_block<N + 1, 2 * N + 1, N, 2 * N>(A, T(-1) * I);
      // (iv) third block assignment s >= 0
      A = linalg::set_block<2 * N + 1, 3 * N + 1, N, 2 * N>(A, T(-1) * I);

      using Dom = linalg::Matrix<T, 1, 2 * N>;
      using Codom = linalg::Matrix<T, 1, 3 * N + 1>;
      return Polyhedron<PolyhedronMeta<Dom, Codom>, Child>(
          A, b, std::get<0>(this->children));
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
          b(i) = r + c_plus_U_inv_t(idx);
        else
          b(i) = r - c_plus_U_inv_t(idx);
      }

      A = linalg::set_block<0, N, 0, N>(A, U_inv);
      A = linalg::set_block<N, 2 * N, 0, N>(A, T(-1) * U_inv);

      using Dom = linalg::Matrix<T, 1, N>;
      using Codom = linalg::Matrix<T, 1, 2 * N>;
      return Polyhedron<PolyhedronMeta<Dom, Codom>, Child>(
          A, b, std::get<0>(this->children));
    }
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * C = { x | ||x - c|| <= r }
   *
   * let y = x - c
   *
   * L2
   * K = { (y, t) | ||y|| <= r t, t >= 0 }
   * [ I 0 ] [ x; t ] + [ -c; 0 ] -> (y, rt)
   * [ 0 r ]
   * ||y||_2 <= r t... scaled second order cone
   *
   * Linf
   * K = { (y, t) | ||y|| <= r t, t >= 0 }
   * -r t <= y_i <= r t \forall i
   * -r t - y_i <= 0 to 0 <= y_i + r t nonneg orthant cone
   * 0 <= r t - y_i... nonneg orthant cone
   *
   * [ I  r ] [ x; t ] + [ -c; c ]
   * [ -I r ]
   *
   * L1
   * K = { (y, t) | ||y|| <= r t, t >= 0 }
   * let u_i >= |y_i|... intoduce first convex constraint
   * sum(u_i) <= r t... second convex constraint (simplex)
   * u_i >= 0... third convex constraint
   * t >= 0... fourth convex constraint
   *
   * -u_i <= y_i <= u_i
   * -u_i - y_i <= 0 to 0 <= y_i + u_i nonneg orthant cone
   * 0 <= u_i - y_i... nonneg orthant cone
   * 1^T u - r t <= 0 to 0 <= r t - 1^T u... nonneg orthant cone
   *
   * [ I  I    0 ] [ x; u; t ] + [ -c; c; 0]
   * [ -I I    0 ]
   * [ 0  -1^T r ]
   */
  template <AtomType AType, BuilderLike Builder>
  constexpr auto canonicalize_set(const Builder& builder) const {
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

    const auto I = linalg::make_identity<T, N>();
    if constexpr (Type == NormBallType::L2) {
      linalg::Matrix<T, N + 1, N + 1> F;
      F = linalg::set_block<0, N, 0, N>(F, I);
      F(N)(N) = r;
      linalg::Vector<T, N + 1> g;
      for (size_t i = 0; i < N; ++i) g(i) = T(-1) * c(i);
      const auto map = affine::AffineMap<decltype(F)>(F, g);
      const auto soc = conic::SecondOrderCone<linalg::Vector<T, N>>();
      const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
      const auto cones = std::make_tuple(soc, rplus);

      const auto new_builder =
          ConeBlockBuilder<builder_dom_t<decltype(builder)>, decltype(map),
                           decltype(child_cb)>(map, child_cb);
      return new_builder.template make<AType>(this->idx, this->lift_map, cones);
    } else if constexpr (Type == NormBallType::Linf) {
      linalg::Matrix<T, 2 * N, N + 1> F;
      F = linalg::set_block<0, N, 0, N>(F, I);
      F = linalg::set_block<N, 2 * N, 0, N>(F, T(-1) * I);
      for (size_t i = 0; i < 2 * N; ++i) F(i)(N) = r;

      linalg::Vector<T, 2 * N> g;
      for (size_t i = 0; i < N; ++i) g(i) = T(-1) * c(i);
      for (size_t i = N, j = 0; i < 2 * N; ++i, ++j) g(i) = c(j);
      const auto map = affine::AffineMap<decltype(F)>(F, g);
      const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
      const auto cones = std::make_tuple(rplus, rplus);

      const auto new_builder =
          ConeBlockBuilder<builder_dom_t<decltype(builder)>, decltype(map),
                           decltype(child_cb)>(map, child_cb);
      return new_builder.template make<AType>(this->idx, this->lift_map, cones);
    } else {
      linalg::Matrix<T, 2 * N + 1, 2 * N + 1> F;
      F = linalg::set_block<0, N, 0, N>(F, I);
      F = linalg::set_block<N, 2 * N, 0, N>(F, T(-1) * I);
      F = linalg::set_block<0, N, N, 2 * N>(F, I);
      F = linalg::set_block<N, 2 * N, N, 2 * N>(F, I);
      for (size_t i = 0; i < N; ++i) F(2 * N)(N + i) = T(-1);
      F(2 * N)(2 * N) = r;
      linalg::Vector<T, 2 * N + 1> g;
      for (size_t i = 0; i < N; ++i) g(i) = T(-1) * c(i);
      for (size_t i = N, j = 0; i < 2 * N; ++i, ++j) g(i) = c(j);
      const auto map = affine::AffineMap<decltype(F)>(F, g);
      const auto rplus1 = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
      const auto rplus2 = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
      const auto cones = std::make_tuple(rplus1, rplus1, rplus2);

      const auto new_builder =
          ConeBlockBuilder<builder_dom_t<decltype(builder)>, decltype(map),
                           decltype(child_cb)>(map, child_cb);
      return new_builder.template make<AType>(this->idx, this->lift_map, cones);
    }
  }

  /** @brief member variables in the norm ball */
  PointT c;  //< n-dim vector of type T representing center of ball
  T r;       //< scalar of type T representing radius
};

}  // namespace convex
