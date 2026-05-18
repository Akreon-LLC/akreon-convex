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

#include <utility>

#include "control/control_traits.hpp"
#include "linalg/linalg_util.hpp"

namespace control {

/**
 * @brief PlantModel
 *
 * Contains state space with input and output port layouts.
 */
template <StateSpaceLike SS, typename InputPorts, typename OutputPorts>
struct PlantModel {
  using StateSpace = linalg::remove_cvref_t<SS>;
  using InputPortsT = linalg::remove_cvref_t<InputPorts>;
  using OutputPortsT = linalg::remove_cvref_t<OutputPorts>;
  using Model = PlantModel<StateSpace, InputPortsT, OutputPortsT>;
  using T = plant_type_t<Model>;

  static constexpr size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr size_t InputDim = plant_traits<Model>::InputDim;
  static constexpr size_t OutputDim = plant_traits<Model>::OutputDim;

  static_assert(InputPortsT::template validates<InputDim>(),
                "Input port layout is invalid for the state-space input");
  static_assert(OutputPortsT::template validates<OutputDim>(),
                "Output port layout is invalid for the state-space output");

  constexpr explicit PlantModel(const StateSpace& ss) : state_space(ss) {}

  constexpr PlantModel(const StateSpace& ss, const InputPortsT&,
                       const OutputPortsT&)
      : state_space(ss) {}

  constexpr auto dynamics_matrix() const { return state_space.A; }

  constexpr auto exogenous_matrix() const {
    return input_matrix<PortRole::EXOGENOUS>();
  }

  constexpr auto control_matrix() const {
    return input_matrix<PortRole::CONTROL>();
  }

  constexpr auto performance_matrix() const {
    return output_matrix<PortRole::PERFORMANCE>();
  }

  constexpr auto measurement_matrix() const {
    return output_matrix<PortRole::MEASUREMENT>();
  }

  /// @brief feedthrough map from selected input role to selected output role
  template <PortRole OutRole, PortRole InRole>
  constexpr auto feedthrough() const {
    constexpr size_t Rows = OutputPortsT::template total_size<OutRole>();
    constexpr size_t Cols = InputPortsT::template total_size<InRole>();
    linalg::Matrix<T, Rows, Cols> result{};
    if (state_space.D.has_value()) {
      copy_feedthrough_ports<OutRole, InRole, 0, 0>(result);
    }
    return result;
  }

  SS state_space; ///< StateSpace model

private:
  /// @brief aggregate all input ports of a role from the full B matrix
  template <PortRole Role>
  constexpr auto input_matrix() const {
    constexpr size_t Cols = InputPortsT::template total_size<Role>();
    linalg::Matrix<T, StateDim, Cols> result{};
    copy_input_ports<Role, 0, 0>(result);
    return result;
  }

  /// @brief aggregate all output ports of a role from the full C matrix
  template <PortRole Role>
  constexpr auto output_matrix() const {
    constexpr size_t Rows = OutputPortsT::template total_size<Role>();
    linalg::Matrix<T, Rows, StateDim> result{};
    copy_output_ports<Role, 0, 0>(result);
    return result;
  }

  /// @brief copy selected B columns into an aggregated input matrix
  template <PortRole Role, size_t Index, size_t DestOffset, typename Mat>
  constexpr void copy_input_ports(Mat& result) const {
    constexpr size_t Count = InputPortsT::template count<Role>();
    if constexpr (Index < Count) {
      constexpr size_t Offset = InputPortsT::template offset<Role, Index>();
      constexpr size_t Size = InputPortsT::template size<Role, Index>();
      const auto block =
          linalg::get_block<0, StateDim, Offset, Offset + Size>(state_space.B);
      result = linalg::set_block<0, StateDim, DestOffset, DestOffset + Size>(
          result, block);
      copy_input_ports<Role, Index + 1, DestOffset + Size>(result);
    }
  }

  /// @brief copy selected C rows into an aggregated output matrix
  template <PortRole Role, size_t Index, size_t DestOffset, typename Mat>
  constexpr void copy_output_ports(Mat& result) const {
    constexpr size_t Count = OutputPortsT::template count<Role>();
    if constexpr (Index < Count) {
      constexpr size_t Offset = OutputPortsT::template offset<Role, Index>();
      constexpr size_t Size = OutputPortsT::template size<Role, Index>();
      const auto block =
          linalg::get_block<Offset, Offset + Size, 0, StateDim>(state_space.C);
      result = linalg::set_block<DestOffset, DestOffset + Size, 0, StateDim>(
          result, block);
      copy_output_ports<Role, Index + 1, DestOffset + Size>(result);
    }
  }

  /// @brief copy selected D row blocks for each matching output port
  template <PortRole OutRole, PortRole InRole, size_t OutIndex,
            size_t OutDestOffset, typename Mat>
  constexpr void copy_feedthrough_ports(Mat& result) const {
    constexpr size_t OutCount = OutputPortsT::template count<OutRole>();
    if constexpr (OutIndex < OutCount) {
      constexpr size_t OutOffset =
          OutputPortsT::template offset<OutRole, OutIndex>();
      constexpr size_t OutSize =
          OutputPortsT::template size<OutRole, OutIndex>();
      copy_feedthrough_input_ports<OutRole, InRole, OutIndex, OutOffset,
                                   OutSize, 0, 0>(result);
      copy_feedthrough_ports<OutRole, InRole, OutIndex + 1,
                             OutDestOffset + OutSize>(result);
    }
  }

  /// @brief copy selected D column blocks for one selected output port
  template <PortRole OutRole, PortRole InRole, size_t OutIndex,
            size_t OutOffset, size_t OutSize, size_t InIndex,
            size_t InDestOffset, typename Mat>
  constexpr void copy_feedthrough_input_ports(Mat& result) const {
    constexpr size_t InCount = InputPortsT::template count<InRole>();
    if constexpr (InIndex < InCount) {
      constexpr size_t InOffset =
          InputPortsT::template offset<InRole, InIndex>();
      constexpr size_t InSize = InputPortsT::template size<InRole, InIndex>();
      constexpr size_t OutDestOffset =
          feedthrough_output_destination<OutRole, OutIndex>();
      const auto block =
          linalg::get_block<OutOffset, OutOffset + OutSize, InOffset,
                            InOffset + InSize>(*state_space.D);
      result =
          linalg::set_block<OutDestOffset, OutDestOffset + OutSize,
                            InDestOffset, InDestOffset + InSize>(result, block);
      copy_feedthrough_input_ports<OutRole, InRole, OutIndex, OutOffset,
                                   OutSize, InIndex + 1, InDestOffset + InSize>(
          result);
    }
  }

  /// @brief aggregated row offset for an output port selected from D
  template <PortRole Role, size_t Index>
  static consteval size_t feedthrough_output_destination() {
    size_t dest = 0;
    [&]<size_t... Is>(std::index_sequence<Is...>) consteval {
      ((Is < Index ? dest += OutputPortsT::template size<Role, Is>() : 0), ...);
    }(std::make_index_sequence<OutputPortsT::template count<Role>()>{});
    return dest;
  }
};

} // namespace control
