/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include "config/assert.hpp"

#include "affine_map.hpp"

namespace affine {

/**
 * @TODO (dakre) add quaternions and SO(3) reps
 */

/**
 * @brief Euler angle rotation about 2d plane
 */
template <typename T>
  requires(linalg::FieldLike<T> && !linalg::is_complex_v<T>)
struct Rotation2d {
  constexpr auto operator()(const T& phi) const {
    const auto cz = linalg::constexpr_cos(phi);
    const auto sz = linalg::constexpr_sin(phi);
    return linalg::Matrix<T, 2, 2>{linalg::Vector<T, 2>{cz, T(-1) * sz},
                                   linalg::Vector<T, 2>{sz, cz}};
  }
};

/**
 * @brief Euler angle rotation about 3d roll, pitch, and yaw angles
 */
template <typename T>
  requires(linalg::FieldLike<T> && !linalg::is_complex_v<T>)
struct Rotation3d {
  constexpr auto x(const T& phi) const {
    const auto cx = linalg::constexpr_cos(phi);
    const auto sx = linalg::constexpr_sin(phi);
    return linalg::Matrix<T, 3, 3>{linalg::Vector<T, 3>{T(1), T(0), T(0)},
                                   linalg::Vector<T, 3>{T(0), cx, T(-1) * sx},
                                   linalg::Vector<T, 3>{T(0), sx, cx}};
  }

  constexpr auto y(const T& theta) const {
    const auto cy = linalg::constexpr_cos(theta);
    const auto sy = linalg::constexpr_sin(theta);
    return linalg::Matrix<T, 3, 3>{linalg::Vector<T, 3>{cy, T(0), sy},
                                   linalg::Vector<T, 3>{T(0), T(1), T(0)},
                                   linalg::Vector<T, 3>{T(-1) * sy, T(0), cy}};
  }

  constexpr auto z(const T& psi) const {
    const auto cz = linalg::constexpr_cos(psi);
    const auto sz = linalg::constexpr_sin(psi);
    return linalg::Matrix<T, 3, 3>{linalg::Vector<T, 3>{cz, T(-1) * sz, T(0)},
                                   linalg::Vector<T, 3>{sz, cz, T(0)},
                                   linalg::Vector<T, 3>{T(0), T(0), T(1)}};
  }
};

/**
 * @brief Scale directions spanning N dim space
 */
template <typename T, size_t N>
  requires(linalg::FieldLike<T> && !linalg::is_complex_v<T> && N > 0)
struct Scale {
  template <typename... Scalars>
  constexpr auto operator()(Scalars... scalars) const {
    auto S = linalg::make_identity<T, N>();
    int i = 0;
    auto set_elem = [&](const auto& s) {
      S(i)(i) *= s;
      i++;
    };
    ((void)set_elem(std::forward<Scalars>(scalars)), ...);
    return S;
  }
};

/**
 * @brief Shear transform where jth component "shears" ith direction
 */
template <typename T, size_t N>
  requires(linalg::FieldLike<T> && !linalg::is_complex_v<T> && N > 0)
struct Shear {
  constexpr auto operator()(const size_t& i, const size_t& j,
                            const T& s) const {
    AK_ASSERT((i < N && j < N) &&
              "Shear transform indices must be within matrix bounds");
    auto H = linalg::make_identity<T, N>();
    H(i)(j) = s;
    return H;
  }
};

} // namespace affine
