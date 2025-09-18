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
template <typename T, size_t N>
using InnerProductSpace = topology::MetricSpace<T, N, decltype(L2<T, N>)>;

}  // namespace linalg
