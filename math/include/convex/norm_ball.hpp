/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"
#include "ellipsoid.hpp"

namespace convex {
/// @brief simple enum to describe ball type
enum struct NormBallType { L2OB, L2CB, L1OB, L1CB, LinfOB, LinfCB };

/**
 * @brief Norm ball defined on valid vector space and one of the following type
 * traits
 *
 * { x | norm(x - c) <= r }... closed norm (L2, L1, Linf) ball that lies within
 * radius { x | norm(x - c) < r}... open "" ball that ""
 */
template <linalg::VectorSpace Space, NormBallType Type>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormBall : public ConvexSet<Space, NormBall<Space, Type>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NormBall<Space, Type>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @briaf alias for scalar type of undering set type
  using ScalarT = linalg::scalar_type_t<T>;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR for NormBall set
   *
   * @param Point (i.e., n-dim vector of type T)
   */
  constexpr explicit NormBall(const Point& c_in, const ScalarT& r_in)
      : c(c_in), r(r_in) {}

 private:
  /// @brief allow base class to invoke impl
  friend ConvexBase;

  /**
   * @brief contains function to see if x is in the norm ball (norm(x - c) <= r)
   *
   * @param Point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    (void)tol;
    using T = linalg::vec_type_t<Point>;
    switch (Type) {
      case NormBallType::L2OB:
        return linalg::L2<T, N>()(x - c) < r;
      case NormBallType::L2CB:
        return linalg::L2<T, N>()(x - c) <= r;
      case NormBallType::L1OB:
        return linalg::L1<T, N>()(x - c) < r;
      case NormBallType::L1CB:
        return linalg::L1<T, N>()(x - c) <= r;
      case NormBallType::LinfOB:
        return linalg::Linf<T, N>()(x - c) < r;
      case NormBallType::LinfCB:
      default:
        return linalg::Linf<T, N>()(x - c) <= r;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex set
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return r > 0; }

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
   *
   * 1. L2 case:
   * Let z = y - (U c + t) \implies ||U^{-1} z||_2^2 <= r^2 \iff z^T (U^-T U^-1)
   * z <= r^2
   * So we have f(Br(x)) = {y | (y - (U c + t))^T (U U^T)^{-1} (y - (U c + 1))
   * <= r^2
   *
   * (i)  c' = U c + t
   * (ii) r' = ||U^{-1}|| \to r \sigma(U) OR U U^T r^2
   *
   * 2. L1 case:
   * We have { y | ||U^{-1} (y - (U c + t))|| <= r }... let z = y - (U c + t)
   * and w = U^{-1} z \to z = U w
   *
   * From above we have ||U^{-1} z || <= r \to ||U^{-1} U w|| <= r simplifies to
   * || w || <= r
   *
   * Linearize L1 norm with abs value epigraphs:
   * || w || <= r \iff \exists s \in R^n_+ s.t. -s <= w <= s, 1^T s <= r
   *
   * L1 ball now represented as a polyhedron (LP equalities & inequalities):
   * P = { y | \exists w, s >= 0 s.t. y - (U c + t) = U w, || w || <= r }
   * x = [y; w; s] where y = f(x), w = U^{-1} z, and s \in R^n_+ s.t. 1^T s <= r
   *
   * ^ extending dim to include w explicitly avoids performing matrix inverse
   * which can be costly (i.e., w = U^{-1}(y - (U c + t))
   *
   * Equalities (Cx = d)
   * (i) y - (U c + t) = U w
   * C x = d
   * where C = [I -U 0], x = [y w s]^T and d = U c + t
   *
   * Inequalities (Ax <= b)
   * (ii) -s <= w <= s ... \implies s >= 0 ||0||_1 undefined
   * w - s <= 0 \implies [0 I -I] x <= 0
   * -w - s <= 0 \implies [0 -I -I] x <= 0
   *
   * (iii) 1^T s <= r
   * 1^T s <= r \implies [0 0 1^T] x <= r
   *
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
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    const auto& U = map.linear_part();
    const auto& t = map.translation();
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
    } else if constexpr (Type == NormBallType::L1OB ||
                         Type == NormBallType::L1CB) {
      if constexpr (linalg::is_complex_v<T>)
        return false;  // TODO -impl
      else {
        // (dakre) TODO - need to test out more
        const auto M = linalg::mat_rows_v<Mat>;
        linalg::Matrix<T, 2 * N + 1, M + 2 * N> A;
        linalg::Vector<T, 2 * N + 1> b;
        b[2 * N] = r;

        const auto I_n = linalg::make_identity<T, N>();
        A = linalg::set_block<0, N, M, M + N>(A, I_n);
        A = linalg::set_block<0, N, M + N, M + 2 * N>(A, T(-1) * I_n);
        A = linalg::set_block<N, 2 * N, M, M + N>(A, T(-1) * I_n);
        A = linalg::set_block<N, 2 * N, M + N, M + 2 * N>(A, T(-1) * I_n);
        for (size_t i = 0; i < N; ++i) A[2 * N][N + M + i] = T(1);

        const auto I_m = linalg::make_identity<T, M>();
        linalg::Matrix<T, M, M + 2 * N> C;
        C = linalg::set_block<0, M, 0, M>(C, I_m);
        C = linalg::set_block<0, M, M, M + N>(C, T(-1) * U);
        const auto d = U * c + t;

        using NewSpace = linalg::Matrix<T, 1, M + 2 * N>;
        if constexpr (Type == NormBallType::L1OB)
          return Polyhedron<NewSpace, HalfspaceType::OLH, decltype(A),
                            decltype(C)>(A, b, C, d);
        else
          return Polyhedron<NewSpace, HalfspaceType::CLH, decltype(A),
                            decltype(C)>(A, b, C, d);
      }
    } else {
      return false;  // TODO -impl
    }
  }

  /** @brief member variables in the norm ball */
  Point c;    //< n-dim vector of type T representing center of ball
  ScalarT r;  //< scalar of type T representing radius
};

}  // namespace convex
