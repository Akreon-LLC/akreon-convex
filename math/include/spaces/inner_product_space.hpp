/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../linalg/linear_algebra.hpp"

namespace spaces {

using namespace linalg;

template <typename T, size_t N>
concept InnerProductSpace =
    requires(const scalar_type_t<Vector<T, N>>& alpha,
             const scalar_type_t<Vector<T, N>>& beta, const Vector<T, N>& X,
             const Vector<T, N>& Y) {
      { alpha * X } -> std::same_as<Vector<T, N>>;
      { alpha * (X + Y) == alpha * X + alpha * Y };
      { (alpha + beta) * X == alpha * X + beta * X };
      { (alpha * beta) * X == alpha * (beta * X) };
      { dot(X, Y) } -> std::same_as<scalar_type_t<T>>;
      { L2(X) >= 0.f };
      { L2(Vector<T, 1>{0.f}) == 0.f };
      { constexpr_abs(alpha) * L2(X) == L2(constexpr_abs(alpha) * X) };
      {
        constexpr_abs(dot(X, Y)) <= L2(X) + L2(Y)
      };                               // cauchy-schwarz inequality
      { L2(X + Y) <= L2(X) + L2(Y) };  // triangle inequality
    };

}  // namespace spaces
