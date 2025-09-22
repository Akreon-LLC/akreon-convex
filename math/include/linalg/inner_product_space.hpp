/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "linear_algebra.hpp"
#include "matrix_exp.hpp"
#include "matrix_inverse.hpp"

namespace linalg {

/**
 * @brief InnerProductLike API requirements
 *
 * concept for constraining a valid inner product space
 */
template <class VectorLike>
concept InnerProductLike =
    FieldLike<typename VectorLike::value_type> &&
    requires(const VectorLike::value_type& alpha,
             const VectorLike::value_type& beta, const VectorLike& x,
             const VectorLike& y) {
      { alpha * x } -> std::same_as<VectorLike>;
      { alpha * (x + y) } -> std::same_as<VectorLike>;
      { alpha * x + alpha * y } -> std::same_as<VectorLike>;
      { (alpha + beta) * x } -> std::same_as<VectorLike>;
      { dot(x, y) } -> std::same_as<typename VectorLike::value_type>;
      {
        L2<typename VectorLike::value_type, VectorLike::s_size>()(x)
      } -> std::same_as<scalar_type_t<typename VectorLike::value_type>>;
      {
        L2<typename VectorLike::value_type, 1>()(
            Vector<typename VectorLike::value_type, 1>(alpha)) *
            L2<typename VectorLike::value_type, VectorLike::s_size>()(x)
      } -> std::same_as<scalar_type_t<typename VectorLike::value_type>>;
      {
        constexpr_abs(dot(x, y))
      } -> std::same_as<typename VectorLike::value_type>;
      {
        L2<typename VectorLike::value_type, VectorLike::s_size>()(x + y)
      } -> std::same_as<typename VectorLike::value_type>;
    };

/**
 * @brief InnerProductSpace alias
 *
 * Inner Product Space axioms
 * 0. alpha * X \exists
 * 1. alpha * (X + Y) == alpha * X + alpha * Y
 * 2. (alpha + beta) * X == alpha * X + beta * X
 * 3. (alpha * beta) * X == alpha * (beta * X)
 * 4. X^T X \exists
 * 5. L2(X) >= 0.
 * 6. |alpha| * L2(X) == L2(|alpha| * X)
 * 7. |X*Y| <= L2(X) + L2(Y)
 * 8. L2(X + Y) <= L2(X) + L2(Y)
 */
template <class VectorLike, size_t M>
  requires(InnerProductLike<VectorLike>)
using InnerProductSpace = topology::MetricSpace<
    VectorLike, M, L2<typename VectorLike::value_type, VectorLike::s_size>>;

}  // namespace linalg
