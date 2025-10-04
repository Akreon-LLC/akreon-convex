/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

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
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormBall : public ConvexSet<Space, NormBall<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NormBall<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @briaf alias for scalar type of undering set type
  using ScalarT = linalg::scalar_type_t<T>;

  /**
   * @brief CTOR for NormBall set
   *
   * @param Point (i.e., n-dim vector of type T)
   */
  constexpr explicit NormBall(const Point& c_in, const ScalarT& r_in,
                              const NormBallType& type_in)
      : c(c_in), r(r_in), type(type_in) {}

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
    constexpr auto N = linalg::vec_size_v<Point>;
    switch (type) {
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
    return other.self().c == c && other.self().r == r &&
           other.self().type == type;
  }

  /** @brief member variables in the norm ball */
  Point c;            //< n-dim vector of type T representing center of ball
  ScalarT r;          //< scalar of type T representing radius
  NormBallType type;  //< holds value of norm ball type
};

}  // namespace convex
