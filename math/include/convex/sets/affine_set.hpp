/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/// @brief forward declaration for AffineMap class definition
template <linalg::MatrixLike Mat>
class AffineMap;

/**
 * @brief AffineSet class definition
 *
 * Base class that holds a template definition for the derived implementations:
 * Parametric forms of an AffineSet via CRTP design.
 */
template <linalg::MatrixLike Mat, typename Derived>
class AffineSet {
 public:
  /// @brief alias to extract underlying type of affine set
  using T = linalg::mat_type_t<Mat>;

  /// @brief helpers to extracting M x N matrix
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;

  /// @brief alias for underlying vector spanning R^n (domain)
  using DomVec = linalg::Vector<T, N>;

  /// @brief alias for underlying vector spanning R^m (codomain)
  using AmbientVec = linalg::Vector<T, M>;

  /// (dakre) TODO refactor to drop this unecessary constraint
  /// @brief static assertion to check that the system is over determined
  static_assert(N >= M);

  /**
   * @brief is_well_formed
   *
   * Checks to see if basis is full column rank
   *
   * @return T/F
   */
  constexpr auto is_well_formed() const { return self().is_well_formed_impl(); }

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
  constexpr auto project(const AmbientVec& x) const {
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
  constexpr auto transform(const AffineMap<Mat>& map) const {
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
  constexpr bool contains(const AmbientVec& x, T tol = 1e-8) const {
    return self().contains_impl(x, tol);
  }

  /**
   * @brief equality comparison overload
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool operator==(const AffineSet<Mat, Derived>& other) const {
    return is_equal(other);
  }

  /**
   * @brief inequality comparison overload
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool operator!=(const AffineSet<Mat, Derived>& other) const {
    return !is_equal(other);
  }

 protected:
  /** @brief member variables in the base class */
  friend Derived;  ///< friend declaration for derived to reference base class
                   ///< member vars
  AmbientVec x0;  ///< reference point that lies in the affine set S \subset R^m

 private:
  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  /**
   * @brief equality comparison
   *
   * @param affine set
   *
   * @return T/F
   */
  constexpr bool is_equal(const AffineSet<Mat, Derived>& other) const {
    const auto basis_equal = all_equal(other.basis(), basis());
    const auto x0_equal = all_equal(other.x0, x0);
    return basis_equal && x0_equal;
  }
};

/**
 * @brief ParametricAffineSet class declaration
 *
 * The set takes the form x0 + span(U1, U2,...Un) where Us are valid column
 * vectors that form a basis for the affine set S = {x0 + U t}
 */
template <linalg::MatrixLike Mat>
class ParametricAffineSet : public AffineSet<Mat, ParametricAffineSet<Mat>> {
 public:
  ///@brief bring into scope the type traits from the base class
  using AffineBase = AffineSet<Mat, ParametricAffineSet<Mat>>;
  using typename AffineBase::AmbientVec;
  using typename AffineBase::DomVec;
  using typename AffineBase::T;

  ///@brief bring into scope the data members (consts)
  using AffineBase::M;
  using AffineBase::N;

  /**
   * @brief CTOR for parametric affine set
   *
   * @param matrix of type T with N x M elements
   * @param vector of type T with M elements
   */
  constexpr explicit ParametricAffineSet(const Mat& U_in,
                                         const AmbientVec& x0_in)
      : U(U_in) {
    this->x0 = x0_in;
  }

  /**
   * @brief is_well_formed_impl
   *
   * Checks to see if the basis is full column rank
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    return linalg::QR(U).get_rank() == M;
  }

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
  constexpr auto project_impl(const DomVec& x) const {
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
  constexpr auto transform_impl(const AffineMap<Mat>& map) const {
    return ParametricAffineSet(
        map.linear_part() * U,
        map.linear_part() * this->x0 + map.translation());
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
  constexpr bool contains_impl(const AmbientVec& x, const T& tol) const {
    const auto r = x - this->x0;
    const auto t = linalg::QR(U).solve(r);
    const auto residual = U * t - r;
    return linalg::L2<T, N>()(residual) < tol;
  }

 private:
  /** @brief member variables in the base class */
  Mat U;  //< basis matrix for the affine set
};

}  // namespace convex
