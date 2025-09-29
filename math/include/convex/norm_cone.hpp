/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "norm_ball.hpp"

namespace convex {

/**
 * @brief Norm cone definition valid on a vector space dim(R^n + 1)
 *
 * { (x, t) | ||x|| <= t }... closed norm cone defined on L1, L2, Linf
 * { (x, t) | ||x|| < t }... open cone defined on L1, L2, Linf
 */
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NormCone : public NormBall<Space> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = NormBall<Space>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief alias for scalar type of underlying set type
  using ScalarT = linalg::scalar_type_t<T>;

  /**
   * @brief CTOR for norm cone
   *
   * @param t (scalar rep variable radius)
   */
  constexpr explicit NormCone(const ScalarT& t_in, const NormBallType& type_in)
      : NormBall<Space>(Point{}, t_in, type_in) {}
};

}  // namespace convex
