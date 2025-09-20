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
    return self().transfrom_impl(map);
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
   * 1. project (x - x0) on basis Q
   * 2. back solve for t (R t = projection)
   * 3. form and return affine projection (x0 + Q * t)
   *
   * @param vector of type T with M elements
   *
   * @return projected vector on this set
   */
  constexpr auto project_impl(const linalg::Vector<T, M>& x) const {
    linalg::Vector<T, N> c;
    if constexpr (linalg::is_complex_v<T>)
      c = linalg::conjugate_transpose(Q) * (x - this->x0);
    else
      c = linalg::transpose(Q) * (x - this->x0);
    const auto t = linalg::back_solve(R, c);
    return this->x0 + Q * t;
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
                               this->x0 + map.transform());
  }

 private:
  linalg::Matrix<T, N, M> U;
};

}  // namespace convex
