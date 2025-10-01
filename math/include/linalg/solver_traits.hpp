/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

namespace linalg {

template <typename T>
struct SolverTraits {
  static constexpr size_t steps = 0;
  static constexpr T tol = T(0);
};

template <>
struct SolverTraits<float> {
  static constexpr size_t steps = 16;
  static constexpr float tol = 1e-8;
};

template <>
struct SolverTraits<double> {
  static constexpr size_t steps = 64;
  static constexpr float tol = 1e-10;
};

}  // namespace linalg
