/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/util.hpp"

namespace convex {

/// @brief forward declaration for AffineMap class definition
template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class AffineMap;

/**
 * @brief AffineSet class definition
 *
 * Base class that holds a template definition for the derived implementations:
 * Implicit or Parametric forms of an AffineSet via CRTP design.
 */
template <typename T, size_t N, size_t M, typename Derived>
  requires(linalg::FieldLike<T> && N >= M)
class AffineSet {
 public:
  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  /**
   * @brief is_valid
   *
   * Checks to see if basis is full column rank
   *
   * @return T/F
   */
  constexpr auto is_valid() const { return self().is_valid_impl(); }

  /**
   * @brief return reference point in set
   *
   * @return x0
   */
  constexpr auto point() const { return x0; }

  /**
   * @brief return basis of affine set
   *
   * @return basis matrix (rep in derived class)
   */
  constexpr auto basis() const { return self().basis_impl(); }

  /**
   * @brief project the vector x onto the affine space
   *
   * @return projection vector
   */
  constexpr auto project(const linalg::Vector<T, M>& x) const {
    return self().project_impl(x);
  }

  /**
   * @brief Transform an affine set by an affine map
   *
   * Given an affine set S = { x | A x = b } and an affine map f(x) = M x + d:
   *
   * 1. **Linear part (M):** Apply the change of basis / linear transform to the
   *    set: S' = { M x | x \in S }.
   *
   * 2. **Translation (d):** Shift the transformed set by d:
   *    S'' = { M x + d | x \in S }.
   *
   * In parametric form, if S = { x0 + U t }, then
   *    f(S) = { (M x0 + d) + (M U) t }.
   *
   * So implementation-wise:
   *   - Apply M to both the reference point (x0) and the basis U.
   *   - Then add d to the transformed reference point.
   *
   * @param vector of type T with N elements (the translation vector d)
   *
   * @return transformed affine set
   */
  constexpr auto transform(const AffineMap<T, N, M>& map) const {
    return self().transform_impl(map);
  }

  /**
   * @brief contains to check if a vector lies in the affine set
   *
   * @param vector x of type T with M elements
   * @param optional tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const linalg::Vector<T, M>& x, T tol = 1e-8) const {
    return self().contains_impl(x, tol);
  }

 private:
  /** @brief member variables in the base class */
  friend Derived;  ///< friend declaration for derived to reference base class
                   ///< member vars
  linalg::Vector<T, M> x0;  ///< reference point that lies in the affine set
};

/**
 * @brief ImplicitAffineSet class definition
 *
 * Derived implementation of AffineSet which is formed by the following set
 * definition: S = {x | A x = b}, thus if A is square and non-singular then this
 * set has one unique solution for x on the hyperplane. If A is singular then x
 * can have many solutions, and its projection error is minimized by the
 * pseudoinverse.
 */
template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class ImplicitAffineSet
    : public AffineSet<T, N, M, ImplicitAffineSet<T, N, M>> {
 public:
  /**
   * @brief CTOR for implicit affine set
   *
   * This will perform QR factorization of A to acquire x0 reference point and
   * stores basis Q, upper right triangular matrix R, and b as member vars in
   * the derived class.
   *
   * @param matrix A of type T with N x M elements
   * @param vector b of type T with N elements
   */
  constexpr explicit ImplicitAffineSet(const linalg::Matrix<T, N, M>& A_in,
                                       const linalg::Vector<T, N>& b_in) {
    const auto qr = linalg::QR(A_in);
    Q = qr.get_Q();
    R = qr.get_R();
    this->x0 = qr.solve(b_in);
    b = b_in;
  }

  /**
   * @brief get basis matrix
   *
   * Returns orthonormal basis which spans the space of ths affine set
   *
   * @return Q
   */
  constexpr auto basis_impl() const { return Q; }

  /**
   * @brief is_valid_impl
   *
   * Checks to see if b \in CS(A) by checking if A x - b == 0
   *
   * @return T/F
   */
  constexpr bool is_valid_impl(T tol = 1e-8) const {
    return L2((Q * R * this->x0) - b) < tol;
  }

  /**
   * @brief project a vector x onto the affine set
   *
   * To do this we need to perform the following steps:
   * 1. Form the residual in the constraint space r = A x - b
   * 2. Compute the left psuedo inverse of A to find the minimum norm solution
   * for x^* (maps constraint residual back into dom(A))... z = A^+ r
   * 3. Apply correction x - Q * z (i.e., projection of x onto the affine set)
   *
   * @param vector of type T with M elements
   *
   * @return projected vector on this set
   */
  constexpr auto project_impl(const linalg::Vector<T, M>& x) const {
    const auto x_proj = (Q * R) * x;
    const auto r = x_proj - b;
    linalg::Vector<T, M> z;
    if constexpr (linalg::is_complex_v<T>)
      z = linalg::inverse(linalg::conjugate_transpose(R)) * r;
    else
      z = linalg::inverse(linalg::transpose(R)) * r;
    return x - Q * z;
  }

  /**
   * @brief transform the affine set by an affine map
   *
   * 1. perform the linear transform to change the basis (i.e., M x)
   * 2. shift the transformed set by b (i.e., M x + b)
   *
   * @param reference to a valid affine map
   *
   * @return transformed affine set
   */
  constexpr auto transform_impl(const AffineMap<T, N, M>& map) const {
    return ImplicitAffineSet(map.linear_part() * (Q * R),
                             b + (Q * R) * map.translation());
  }

  /**
   * @brief contains function to see if x is in the affine set
   *
   * @param vector x of type T with M elements
   * @param tolerance scalar
   *
   * @return T/F
   */
  constexpr bool contains_impl(const linalg::Vector<T, M>& x,
                               const T& tol) const {
    const auto residual = (Q * R) * x - b;
    return linalg::L2(residual) < tol;
  }

 private:
  /** @brief member variables in the base class */
  linalg::Matrix<T, N, N>
      Q;  //< Q basis matrix spanning the space of the affine set
  linalg::Matrix<T, N, M>
      R;  //< R upper right triangular matrix factored from QR(A)
  linalg::Vector<T, N> b;  //< b bias vector which lives in the CS(A)
};

/**
 * @brief ParametricAffineSet class declaration
 *
 * The set takes the form x0 + span(U1, U2,...Un) where Us are valid column
 * vectors that form a basis for the affine set S = {x0 + U t}
 */
template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class ParametricAffineSet
    : public AffineSet<T, N, M, ParametricAffineSet<T, N, M>> {
 public:
  /**
   * @brief CTOR for parametric affine set
   *
   * @param matrix of type T with N x M elements
   * @param vector of type T with M elements
   */
  constexpr explicit ParametricAffineSet(const linalg::Matrix<T, N, M>& U_in,
                                         const linalg::Vector<T, M> x0_in)
      : U(U_in) {
    this->x0 = x0_in;
  }

  /**
   * @brief is_valid_impl
   *
   * Checks to see if the basis is full column rank
   *
   * @return T/F
   */
  constexpr bool is_valid_impl() const { return linalg::QR(U).get_rank() == M; }

  /**
   * @brief simple basis return
   *
   * @return U
   */
  constexpr auto basis_impl() const { return U; }

  /**
   * @brief project x onto the affine set
   *
   * This one is much simpler than S = {x | Ax = b}, we just perform QR
   * factorization on U and back solve for the residual vector (x - x0) which
   * gives us a t vector. The t vector is then projected onto the basis U and
   * added by the reference point x0 to obtain the new projection
   *
   * @param vector of type T M elements
   *
   * @return projected x on affine set
   */
  constexpr auto project_impl(const linalg::Vector<T, M>& x) const {
    const auto t = linalg::QR(U).solve(x - this->x0);
    return this->x0 + U * t;
  }

  /**
   * @brief transform affine set by map
   *
   * This one is also more simple than implicit case, we just simply perform the
   * change of basis on U, and then apply the translation directly to the
   * reference point.
   *
   * @param reference to a valid affine map
   *
   * @return parametric affine set object
   */
  constexpr auto transform_impl(const AffineMap<T, N, M>& map) const {
    return ParametricAffineSet(map.linear_part() * U,
                               this->x0 + map.translation());
  }

  /**
   * @brief contains
   *
   * Simple check to see if the vector lies in the affine set as defined by the
   * parametric set
   *
   * @param vector of type T with M elements
   * @param tolerance scalar
   *
   * @return T/F
   */
  constexpr bool contains_impl(const linalg::Vector<T, M>& x,
                               const T& tol) const {
    const auto r = x - this->x0;
    const auto t = linalg::QR(U).solve(r);
    const auto residual = U * t - r;
    return linalg::L2(residual) < tol;
  }

 private:
  /** @brief member variables in the base class */
  linalg::Matrix<T, N, M> U;  //< basis matrix for the affine set
};

}  // namespace convex
