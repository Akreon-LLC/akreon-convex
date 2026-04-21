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

/// @brief forward decl of state space
template <linalg::SquareMatrixLike PlantIn, linalg::MatrixLike ControlIn,
          linalg::MatrixLike OutputIn>
struct StateSpace;

/// @brief base type traits for state space
template <class StateSpace>
struct ss_traits;

/// @brief explicit type traits for state space
template <linalg::SquareMatrixLike PlantIn, linalg::MatrixLike ControlIn,
          linalg::MatrixLike OutputIn>
struct ss_traits<StateSpace<PlantIn, ControlIn, OutputIn>> {
  using Plant = linalg::remove_cvref_t<PlantIn>;
  using Control = linalg::remove_cvref_t<ControlIn>;
  using Output = linalg::remove_cvref_t<OutputIn>;

  using T = linalg::mat_type_t<Plant>;
  using State = linalg::Vector<T, linalg::mat_cols_v<Plant>>;
  using Input = linalg::Vector<T, linalg::mat_cols_v<Control>>;
  using OutVec = linalg::Vector<T, linalg::mat_rows_v<Output>>;
};

/// @brief extract scalar type from ss
template <typename StateSpace>
using ss_type_t = typename ss_traits<linalg::remove_cvref_t<StateSpace>>::T;

/// @brief ss plant type
template <typename StateSpace>
using ss_plant_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Plant;

/// @brief ss control type
template <typename StateSpace>
using ss_control_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Control;

/// @brief ss output type
template <typename StateSpace>
using ss_output_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Output;

/// @brief extract state vector
template <typename StateSpace>
using state_vec_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::State;

/// @brief extract input vector
template <typename StateSpace>
using input_vec_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::Input;

/// @brief extract output vector
template <typename StateSpace>
using output_vec_t =
    typename ss_traits<linalg::remove_cvref_t<StateSpace>>::OutVec;

/// @brief StateSpace concept
template <typename StateSpace>
concept StateSpaceLike = requires(const StateSpace& ss) {
  { ss.A };
  { ss.B };
  { ss.C };
};

/// @brief compatibility concept retained until synthesis is migrated
template <typename StateSpace>
concept ContinuousStateSpaceLike = StateSpaceLike<StateSpace>;

/// @brief compatibility concept retained until synthesis is migrated
template <typename StateSpace>
concept DiscreteStateSpaceLike = StateSpaceLike<StateSpace>;

}  // namespace control
