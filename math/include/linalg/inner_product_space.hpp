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
template <class Vec>
concept InnerProductLike =
    VectorLike<Vec> &&
    requires(const Vec::value_type& alpha, const Vec::value_type& beta,
             const Vec& x, const Vec& y) {
      { alpha * x } -> std::same_as<Vec>;
      { alpha * (x + y) } -> std::same_as<Vec>;
      { alpha * x + alpha * y } -> std::same_as<Vec>;
      { (alpha + beta) * x } -> std::same_as<Vec>;
      { dot(x, y) } -> std::same_as<typename Vec::value_type>;
      { L2<Vec>()(x) } -> std::same_as<scalar_type_t<typename Vec::value_type>>;
      {
        L2<Vector<typename Vec::value_type, 1>>()(
            Vector<typename Vec::value_type, 1>(alpha)) *
            L2<Vec>()(x)
      } -> std::same_as<scalar_type_t<typename Vec::value_type>>;
      { constexpr_abs(dot(x, y)) } -> std::same_as<typename Vec::value_type>;
      { L2<Vec>()(x + y) } -> std::same_as<typename Vec::value_type>;
    };

}  // namespace linalg
