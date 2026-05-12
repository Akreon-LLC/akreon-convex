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
 * Contains state space with layout of signals
 */
template <StateSpaceLike SS, typename InputLayout, typename OutputLayout>
struct PlantModel {
  using StateSpace = linalg::remove_cvref_t<SS>;
  using InputLayoutT = linalg::remove_cvref_t<InputLayout>;
  using OutputLayoutT = linalg::remove_cvref_t<OutputLayout>;
  using Model = PlantModel<StateSpace, InputLayoutT, OutputLayoutT>;
  using T = plant_type_t<Model>;

  static constexpr size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr size_t InputDim = plant_traits<Model>::InputDim;
  static constexpr size_t OutputDim = plant_traits<Model>::OutputDim;

  static_assert(InputLayoutT::template validates<InputDim>(),
                "Input channel layout is invalid for the state-space input");
  static_assert(OutputLayoutT::template validates<OutputDim>(),
                "Output channel layout is invalid for the state-space output");

  constexpr explicit PlantModel(const StateSpace& ss) : state_space(ss) {}

  constexpr PlantModel(const StateSpace& ss, const InputLayoutT&,
                       const OutputLayoutT&)
      : state_space(ss) {}

  constexpr auto control_matrix() const {
    return input_matrix<InputKind::control>();
  }

  constexpr auto disturbance_matrix() const {
    return input_matrix<InputKind::disturbance>();
  }

  constexpr auto reference_matrix() const {
    return input_matrix<InputKind::reference>();
  }

  constexpr auto process_noise_matrix() const {
    return input_matrix<InputKind::process_noise>();
  }

  constexpr auto measurement_noise_matrix() const {
    return input_matrix<InputKind::measurement_noise>();
  }

  constexpr auto exogenous_matrix() const {
    return input_matrix<InputKind::exogenous>();
  }

  constexpr auto measurement_matrix() const {
    return output_matrix<OutputKind::measurement>();
  }

  constexpr auto performance_matrix() const {
    return output_matrix<OutputKind::performance>();
  }

  constexpr auto regulated_matrix() const {
    return output_matrix<OutputKind::regulated>();
  }

  constexpr auto state_output_matrix() const {
    return output_matrix<OutputKind::state>();
  }

  /// @brief feedthrough map from selected input kind to selected output kind
  template <OutputKind OutKind, InputKind InKind>
  constexpr auto feedthrough() const {
    constexpr size_t Rows = OutputLayoutT::template total_size<OutKind>();
    constexpr size_t Cols = InputLayoutT::template total_size<InKind>();
    linalg::Matrix<T, Rows, Cols> result{};
    if (state_space.D.has_value()) {
      copy_feedthrough_channels<OutKind, InKind, 0, 0, Rows>(result);
    }
    return result;
  }

  SS state_space; ///< StateSpace model

private:
  /// @brief aggregate all input channels of a kind from the full B matrix
  template <InputKind Kind>
  constexpr auto input_matrix() const {
    constexpr size_t Cols = InputLayoutT::template total_size<Kind>();
    linalg::Matrix<T, StateDim, Cols> result{};
    copy_input_channels<Kind, 0, 0>(result);
    return result;
  }

  /// @brief aggregate all output channels of a kind from the full C matrix
  template <OutputKind Kind>
  constexpr auto output_matrix() const {
    constexpr size_t Rows = OutputLayoutT::template total_size<Kind>();
    linalg::Matrix<T, Rows, StateDim> result{};
    copy_output_channels<Kind, 0, 0>(result);
    return result;
  }

  /// @brief copy selected B columns into an aggregated input matrix
  template <InputKind Kind, size_t Index, size_t DestOffset, typename Mat>
  constexpr void copy_input_channels(Mat& result) const {
    constexpr size_t Count = InputLayoutT::template count<Kind>();
    if constexpr (Index < Count) {
      constexpr size_t Offset = InputLayoutT::template offset<Kind, Index>();
      constexpr size_t Size = InputLayoutT::template size<Kind, Index>();
      const auto block =
          linalg::get_block<0, StateDim, Offset, Offset + Size>(state_space.B);
      // DestOffset accumulates sizes of earlier matching channels.
      result = linalg::set_block<0, StateDim, DestOffset, DestOffset + Size>(
          result, block);
      copy_input_channels<Kind, Index + 1, DestOffset + Size>(result);
    }
  }

  /// @brief copy selected C rows into an aggregated output matrix
  template <OutputKind Kind, size_t Index, size_t DestOffset, typename Mat>
  constexpr void copy_output_channels(Mat& result) const {
    constexpr size_t Count = OutputLayoutT::template count<Kind>();
    if constexpr (Index < Count) {
      constexpr size_t Offset = OutputLayoutT::template offset<Kind, Index>();
      constexpr size_t Size = OutputLayoutT::template size<Kind, Index>();
      const auto block =
          linalg::get_block<Offset, Offset + Size, 0, StateDim>(state_space.C);
      // DestOffset accumulates sizes of earlier matching channels.
      result = linalg::set_block<DestOffset, DestOffset + Size, 0, StateDim>(
          result, block);
      copy_output_channels<Kind, Index + 1, DestOffset + Size>(result);
    }
  }

  /// @brief copy selected D row blocks for each matching output channel
  template <OutputKind OutKind, InputKind InKind, size_t OutIndex,
            size_t OutDestOffset, size_t Rows, typename Mat>
  constexpr void copy_feedthrough_channels(Mat& result) const {
    constexpr size_t OutCount = OutputLayoutT::template count<OutKind>();
    if constexpr (OutIndex < OutCount) {
      constexpr size_t OutOffset =
          OutputLayoutT::template offset<OutKind, OutIndex>();
      constexpr size_t OutSize =
          OutputLayoutT::template size<OutKind, OutIndex>();
      copy_feedthrough_input_channels<OutKind, InKind, OutIndex, OutOffset,
                                      OutSize, 0, 0>(result);
      // Advance to the next selected output-channel row block.
      copy_feedthrough_channels<OutKind, InKind, OutIndex + 1,
                                OutDestOffset + OutSize, Rows>(result);
    }
  }

  /// @brief copy selected D column blocks for one selected output channel
  template <OutputKind OutKind, InputKind InKind, size_t OutIndex,
            size_t OutOffset, size_t OutSize, size_t InIndex,
            size_t InDestOffset, typename Mat>
  constexpr void copy_feedthrough_input_channels(Mat& result) const {
    constexpr size_t InCount = InputLayoutT::template count<InKind>();
    if constexpr (InIndex < InCount) {
      constexpr size_t InOffset =
          InputLayoutT::template offset<InKind, InIndex>();
      constexpr size_t InSize = InputLayoutT::template size<InKind, InIndex>();
      constexpr size_t OutDestOffset =
          feedthrough_output_destination<OutKind, OutIndex>();
      const auto block =
          linalg::get_block<OutOffset, OutOffset + OutSize, InOffset,
                            InOffset + InSize>(*state_space.D);
      result =
          linalg::set_block<OutDestOffset, OutDestOffset + OutSize,
                            InDestOffset, InDestOffset + InSize>(result, block);
      // Advance to the next selected input-channel column block.
      copy_feedthrough_input_channels<OutKind, InKind, OutIndex, OutOffset,
                                      OutSize, InIndex + 1,
                                      InDestOffset + InSize>(result);
    }
  }

  /// @brief aggregated row offset for an output channel selected from D
  template <OutputKind Kind, size_t Index>
  static consteval size_t feedthrough_output_destination() {
    size_t dest = 0;
    // Sum sizes of selected output channels that precede Index.
    [&]<size_t... Is>(std::index_sequence<Is...>) consteval {
      ((Is < Index ? dest += OutputLayoutT::template size<Kind, Is>() : 0),
       ...);
    }(std::make_index_sequence<OutputLayoutT::template count<Kind>()>{});
    return dest;
  }
};

} // namespace control
