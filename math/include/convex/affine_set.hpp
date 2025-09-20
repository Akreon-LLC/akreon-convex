/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/util.hpp"

namespace convex {

template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class AffineMap;

template <typename T, size_t N, size_t M, typename Derived>
  requires(linalg::FieldLike<T> && M >= N)
class AffineSet {
 public:
  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  constexpr auto point() const { return x0; }

  constexpr auto basis() const { return self().basis_impl(); }

  constexpr auto project(const linalg::Vector<T, M>& x) const {
    return self().project_impl(x);
  }

  constexpr auto transform(const AffineMap<T, N, M>& map) const {
    return self().transform_impl(map);
  }

  constexpr bool contains(const linalg::Vector<T, M>& x, T tol = 1e-8) const {
    return self().contains_impl(x, tol);
  }

 private:
  friend Derived;
  linalg::Vector<T, M> x0;
};

template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class ImplicitAffineSet
    : public AffineSet<T, N, M, ImplicitAffineSet<T, N, M>> {
 public:
  constexpr explicit ImplicitAffineSet(const linalg::Matrix<T, N, M>& A_in,
                                       const linalg::Vector<T, N>& b_in) {
    const auto qr = linalg::QR(A_in);
    Q = qr.get_Q();
    R = qr.get_R();
    this->x0 = qr.solve(b_in);
    b = b_in;
  }

  constexpr auto basis_impl() const { return Q; }

  /**
   * @brief project a vector x onto the column space of the basis spanning the
   * affine set
   *
   * 1. Form the residual in the constraint space r = A x - b
   * 2. Compute the left psuedo inverse of A to find the minimum norm solution
   * for x^* (maps constraint residual back into dom(A))
   * (a) z = A^+ r
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
   * @brief translate affine set by vector d
   *
   * 1. S = {x | Ax = b} \to S + d = {x + d | x \in S}
   *
   * @param vector of type T with N elements
   *
   * @return translated affine set
   */
  constexpr auto transform_impl(const AffineMap<T, N, M>& map) const {
    return ImplicitAffineSet(map.linear_part() * (Q * R),
                             b + (Q * R) * map.translation());
  }

  constexpr bool contains_impl(const linalg::Vector<T, M>& x,
                               const T& tol) const {
    const auto residual = (Q * R) * x - b;
    return linalg::L2(residual) < tol;
  }

 private:
  linalg::Matrix<T, N, N> Q;
  linalg::Matrix<T, N, M> R;
  linalg::Vector<T, N> b;
};

template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class ParametricAffineSet
    : public AffineSet<T, N, M, ParametricAffineSet<T, N, M>> {
 public:
  constexpr explicit ParametricAffineSet(const linalg::Matrix<T, N, M>& U_in,
                                         const linalg::Vector<T, M> x0_in)
      : U(U_in) {
    this->x0 = x0_in;
  }

  constexpr auto basis_impl() const { return U; }

  constexpr auto project_impl(const linalg::Vector<T, M>& x) const {
    const auto t = linalg::QR(U).solve(x - this->x0);
    return this->x0 + U * t;
  }

  constexpr auto transform_impl(const AffineMap<T, N, M>& map) const {
    return ParametricAffineSet(map.linear_part() * U,
                               this->x0 + map.translation());
  }

  constexpr bool contains_impl(const linalg::Vector<T, M>& x,
                               const T& tol) const {
    const auto r = x - this->x0;
    const auto t = linalg::QR(U).solve(r);
    const auto residual = U * t - r;
    return linalg::L2(residual) < tol;
  }

 private:
  linalg::Matrix<T, N, M> U;
};

}  // namespace convex
