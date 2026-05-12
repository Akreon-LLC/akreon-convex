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

#include "control/control_traits.hpp"

namespace control {

/// @brief plant dynamics matrix for state-space or plant model
template <ContinuousStateSpaceLike SS>
constexpr const auto& plant_A(const SS& ss) {
  return ss.A;
}

/// @brief plant dynamics matrix for state-space or plant model
template <ContinuousPlantModelLike Plant>
constexpr const auto& plant_A(const Plant& plant) {
  return plant.state_space.A;
}

/// @brief control input matrix for state-feedback synthesis
template <ContinuousStateSpaceLike SS>
constexpr const auto& state_feedback_Bu(const SS& ss) {
  return ss.B;
}

/// @brief control input matrix for state-feedback synthesis
template <ContinuousPlantModelLike Plant>
constexpr auto state_feedback_Bu(const Plant& plant) {
  static_assert(plant_traits<linalg::remove_cvref_t<Plant>>::ControlDim > 0,
                "PlantModel state feedback requires a control input channel");
  return plant.control_matrix();
}

/// @brief measurement output matrix for observer synthesis
template <ContinuousStateSpaceLike SS>
constexpr const auto& observer_Cy(const SS& ss) {
  return ss.C;
}

/// @brief measurement output matrix for observer synthesis
template <ContinuousPlantModelLike Plant>
constexpr auto observer_Cy(const Plant& plant) {
  static_assert(plant_traits<linalg::remove_cvref_t<Plant>>::MeasurementDim > 0,
                "PlantModel observer synthesis requires a measurement output "
                "channel");
  return plant.measurement_matrix();
}

} // namespace control
