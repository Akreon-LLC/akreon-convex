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

#include "linalg/vector/algebra_traits.hpp"

namespace control {

/// @brief time domain type
enum struct TimeDomainType : uint8_t { CONTINUOUS = 0, DISCRETE };

/// @brief forward decl of state space
template <linalg::MatrixLike Plant, linalg::MatrixLike Control,
          linalg::MatrixLike Output, linalg::MatrixLike Feedthrough,
          TimeDomainType Type>
struct StateSpace;

/// @brief base type traits for state space
template <class StateSpace>
struct ss_traits;

/// @brief explicit type traits for state space
template <linalg::MatrixLike Plant, linalg::MatrixLike Control,
          linalg::MatrixLike Output, linalg::MatrixLike Feedthrough,
          TimeDomainType Type>
struct ss_traits<StateSpace<Plant, Control, Output, Feedthrough, Type>> {
  using T = linalg::mat_type_t<Plant>;
  static constexpr size_t PlantCols = linalg::mat_cols_v<Plant>;
  static constexpr size_t ControlCols = linalg::mat_cols_v<Control>;
  using State = linalg::Vector<T, PlantCols>;
  using Input = linalg::Vector<T, ControlCols>;
  static constexpr TimeDomainType DType = Type;
};

/// @brief extract state vector
template <typename StateSpace>
using StateVec = typename ss_traits<StateSpace>::State;

/// @brief extract state vector
template <typename StateSpace>
using InputVec = typename ss_traits<StateSpace>::Input;

/// @brief check if continuous or not
template <typename StateSpace>
inline constexpr bool is_continuous_v =
    ss_traits<StateSpace>::DType == TimeDomainType::CONTINUOUS;

/// @brief StateSpace concept
template <typename StateSpace>
concept StateSpaceLike = requires(const StateSpace& ss) {
  { ss.A };
  { ss.B };
  { ss.C };
};

/// @brief ContinuousStateSpace concept
template <typename StateSpace>
concept ContinuousStateSpaceLike =
    is_continuous_v<StateSpace> && StateSpaceLike<StateSpace>;

/// @brief ContinuousStateSpace concept
template <typename StateSpace>
concept DiscreteStateSpaceLike =
    !is_continuous_v<StateSpace> && StateSpaceLike<StateSpace>;

}  // namespace control
