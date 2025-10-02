/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/// @brief simple enum to describe type of halfspace
enum struct HalfspaceType { CLH, OLH, CUH, OUH };

/**
 * @brief Halfspace defined on a valid vector space and one of the following
 * type traits:
 *
 * {x | a^T x <= b}... closed lower halfspace
 * {x | a^T x < b}... open lower halfspace
 * {x | a^T x >= b}... closed upper halfspace
 * {x | a^T x > b}... open upper halfspace
 */
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class Halfspace : public ConvexSet<Space, Halfspace<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, Halfspace<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief halfspace can only be defined over reals
  static_assert(!linalg::is_complex_v<T>);

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /**
   * @brief CTOR for halfspace set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param b bias scalar
   * @param enumer type describing halfspace
   */
  constexpr explicit Halfspace(const Point& a_in, const T& b_in,
                               const HalfspaceType& type_in)
      : a(a_in), b(b_in), type(type_in) {}

  /**
   * @brief contains function to see if x is in the halfspace a^T x <= b
   *
   * @param point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    (void)tol;
    switch (type) {
      case HalfspaceType::OUH:
        return linalg::dot(a, x) > b;
      case HalfspaceType::CUH:
        return linalg::dot(a, x) >= b;
      case HalfspaceType::OLH:
        return linalg::dot(a, x) < b;
      case HalfspaceType::CLH:
      default:
        return linalg::dot(a, x) <= b;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex set
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const {
    if (linalg::all_zero(a)) {
      if (type == HalfspaceType::CLH || type == HalfspaceType::OLH) {
        if (b < T(0)) return false;
      } else {
        if (b > T(0)) return false;
      }
    }
    return true;
  }

 private:
  /** @brief member variables in the halfspace */
  Point a;             //< n-dim vector of type T
  T b;                 //< bias scalar of type T
  HalfspaceType type;  //< holds value of halfspace enum type
};

}  // namespace convex
