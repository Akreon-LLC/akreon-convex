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

#include <concepts>
#include <cstdint>
#include <tuple>

#include "linalg/vector/algebra_traits.hpp"

namespace control {

/// @brief InputKind
enum struct InputKind : uint8_t {
  control,
  disturbance,
  reference,
  process_noise,
  measurement_noise,
  exogenous
};

/// @brief OutputKind
enum struct OutputKind : uint8_t { measurement, performance, regulated, state };

/// @brief Input channel specification
template <InputKind Kind, size_t Offset, size_t Size>
struct InputChannel {
  static constexpr InputKind kind = Kind;
  static constexpr size_t offset = Offset;
  static constexpr size_t size = Size;
};

/// @brief Output channel specification
template <OutputKind Kind, size_t Offset, size_t Size>
struct OutputChannel {
  static constexpr OutputKind kind = Kind;
  static constexpr size_t offset = Offset;
  static constexpr size_t size = Size;
};

template <typename Channel>
concept InputChannelLike = requires {
  { Channel::kind } -> std::convertible_to<InputKind>;
  { Channel::offset } -> std::convertible_to<size_t>;
  { Channel::size } -> std::convertible_to<size_t>;
};

template <typename Channel>
concept OutputChannelLike = requires {
  { Channel::kind } -> std::convertible_to<OutputKind>;
  { Channel::offset } -> std::convertible_to<size_t>;
  { Channel::size } -> std::convertible_to<size_t>;
};

/// @brief Input layout
template <InputChannelLike... Channels>
struct InputLayout {
  using ChannelsTuple = std::tuple<Channels...>;

  /// @brief count channels matching a signal kind
  template <InputKind Kind>
  static consteval size_t count() {
    // Fold over every channel and add one only when the channel kind matches.
    return (size_t{0} + ... + (Channels::kind == Kind ? size_t{1} : size_t{0}));
  }

  /// @brief total dimension for channels matching a signal kind
  template <InputKind Kind>
  static consteval size_t total_size() {
    // Fold over every channel and add its size only when the kind matches.
    return (size_t{0} + ... + (Channels::kind == Kind ? Channels::size : 0));
  }

  /// @brief offset of the Index-th channel matching a signal kind
  template <InputKind Kind, size_t Index>
  static consteval size_t offset() {
    size_t current = 0;
    size_t result = 0;
    // Walk matching channels left-to-right. current is incremented only for
    // matching channels, so current == Index is true for exactly one match.
    ((Channels::kind == Kind
          ? ((current == Index ? (result = Channels::offset, 0) : 0), ++current,
             0)
          : 0),
     ...);
    static_assert(Index < count<Kind>(), "Requested input channel is missing");
    return result;
  }

  /// @brief size of the Index-th channel matching a signal kind
  template <InputKind Kind, size_t Index>
  static consteval size_t size() {
    size_t current = 0;
    size_t result = 0;
    // Same matching-channel walk as offset(), but records channel size.
    ((Channels::kind == Kind
          ? ((current == Index ? (result = Channels::size, 0) : 0), ++current,
             0)
          : 0),
     ...);
    static_assert(Index < count<Kind>(), "Requested input channel is missing");
    return result;
  }

  /// @brief validate channel bounds and overlap against full input dimension
  template <size_t FullInputDim>
  static consteval bool validates() {
    // First fold checks every channel has positive size and in-bounds extent.
    bool ok = ((Channels::size > 0 &&
                Channels::offset + Channels::size <= FullInputDim) &&
               ...);
    bool occupied[FullInputDim == 0 ? 1 : FullInputDim]{};
    auto mark = [&]<typename Channel>() consteval {
      if constexpr (Channel::offset + Channel::size > FullInputDim) { return; }
      // Mark each occupied input index; seeing an index twice means overlap.
      for (size_t i = Channel::offset; i < Channel::offset + Channel::size;
           ++i) {
        if (occupied[i]) ok = false;
        occupied[i] = true;
      }
    };
    // Invoke mark once per channel type in declaration order.
    (mark.template operator()<Channels>(), ...);
    return ok;
  }
};

/// @brief Output Layout
template <OutputChannelLike... Channels>
struct OutputLayout {
  using ChannelsTuple = std::tuple<Channels...>;

  /// @brief count channels matching a signal kind
  template <OutputKind Kind>
  static consteval size_t count() {
    // Fold over every channel and add one only when the channel kind matches.
    return (size_t{0} + ... + (Channels::kind == Kind ? size_t{1} : size_t{0}));
  }

  /// @brief total dimension for channels matching a signal kind
  template <OutputKind Kind>
  static consteval size_t total_size() {
    // Fold over every channel and add its size only when the kind matches.
    return (size_t{0} + ... + (Channels::kind == Kind ? Channels::size : 0));
  }

  /// @brief offset of the Index-th channel matching a signal kind
  template <OutputKind Kind, size_t Index>
  static consteval size_t offset() {
    size_t current = 0;
    size_t result = 0;
    // Walk matching channels left-to-right. current is incremented only for
    // matching channels, so current == Index is true for exactly one match.
    ((Channels::kind == Kind
          ? ((current == Index ? (result = Channels::offset, 0) : 0), ++current,
             0)
          : 0),
     ...);
    static_assert(Index < count<Kind>(), "Requested output channel is missing");
    return result;
  }

  /// @brief size of the Index-th channel matching a signal kind
  template <OutputKind Kind, size_t Index>
  static consteval size_t size() {
    size_t current = 0;
    size_t result = 0;
    // Same matching-channel walk as offset(), but records channel size.
    ((Channels::kind == Kind
          ? ((current == Index ? (result = Channels::size, 0) : 0), ++current,
             0)
          : 0),
     ...);
    static_assert(Index < count<Kind>(), "Requested output channel is missing");
    return result;
  }

  /// @brief validate channel bounds and overlap against full output dimension
  template <size_t FullOutputDim>
  static consteval bool validates() {
    // First fold checks every channel has positive size and in-bounds extent.
    bool ok = ((Channels::size > 0 &&
                Channels::offset + Channels::size <= FullOutputDim) &&
               ...);
    bool occupied[FullOutputDim == 0 ? 1 : FullOutputDim]{};
    auto mark = [&]<typename Channel>() consteval {
      if constexpr (Channel::offset + Channel::size > FullOutputDim) { return; }
      // Mark each occupied output index; seeing an index twice means overlap.
      for (size_t i = Channel::offset; i < Channel::offset + Channel::size;
           ++i) {
        if (occupied[i]) ok = false;
        occupied[i] = true;
      }
    };
    // Invoke mark once per channel type in declaration order.
    (mark.template operator()<Channels>(), ...);
    return ok;
  }
};

/// @brief time domain type
enum struct TimeDomainType : uint8_t { CONTINUOUS = 0, DISCRETE };

/// @brief forward decl of state space
template <TimeDomainType Type, linalg::SquareMatrixLike PlantIn,
          linalg::MatrixLike ControlIn, linalg::MatrixLike OutputIn>
struct StateSpace;

/// @brief base type traits for state space
template <class StateSpace>
struct ss_traits;

/// @brief explicit type traits for state space
template <TimeDomainType Type, linalg::SquareMatrixLike PlantIn,
          linalg::MatrixLike ControlIn, linalg::MatrixLike OutputIn>
struct ss_traits<StateSpace<Type, PlantIn, ControlIn, OutputIn>> {
  using Plant = linalg::remove_cvref_t<PlantIn>;
  using Control = linalg::remove_cvref_t<ControlIn>;
  using Output = linalg::remove_cvref_t<OutputIn>;

  using T = linalg::mat_type_t<Plant>;
  using State = linalg::Vector<T, linalg::mat_cols_v<Plant>>;
  using Input = linalg::Vector<T, linalg::mat_cols_v<Control>>;
  using OutVec = linalg::Vector<T, linalg::mat_rows_v<Output>>;
  static constexpr TimeDomainType DType = Type;
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

/// @brief check if continuous or not
template <typename StateSpace>
inline constexpr bool is_continuous_v =
    ss_traits<linalg::remove_cvref_t<StateSpace>>::DType ==
    TimeDomainType::CONTINUOUS;

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
    StateSpaceLike<StateSpace> && is_continuous_v<StateSpace>;

/// @brief ContinuousStateSpace concept
template <typename StateSpace>
concept DiscreteStateSpaceLike =
    StateSpaceLike<StateSpace> && !is_continuous_v<StateSpace>;

/// @brief Forward declaration of PlantModel
template <StateSpaceLike SS, typename InputLayout, typename OutputLayout>
struct PlantModel;

/// @brief base type traits for plant model
template <class Plant>
struct plant_traits;

/// @brief explicit type traits for plant model
template <StateSpaceLike SS, typename InputLayoutT, typename OutputLayoutT>
struct plant_traits<PlantModel<SS, InputLayoutT, OutputLayoutT>> {
  using StateSpace = linalg::remove_cvref_t<SS>;
  using InputLayout = linalg::remove_cvref_t<InputLayoutT>;
  using OutputLayout = linalg::remove_cvref_t<OutputLayoutT>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;
  using Output = output_vec_t<StateSpace>;
  static constexpr TimeDomainType DType = ss_traits<StateSpace>::DType;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;
  static constexpr size_t OutputDim = linalg::vec_size_v<Output>;

  static constexpr size_t ControlDim =
      InputLayout::template total_size<InputKind::control>();
  static constexpr size_t DisturbanceDim =
      InputLayout::template total_size<InputKind::disturbance>();
  static constexpr size_t ReferenceDim =
      InputLayout::template total_size<InputKind::reference>();
  static constexpr size_t ProcessNoiseDim =
      InputLayout::template total_size<InputKind::process_noise>();
  static constexpr size_t MeasurementNoiseDim =
      InputLayout::template total_size<InputKind::measurement_noise>();
  static constexpr size_t ExogenousDim =
      InputLayout::template total_size<InputKind::exogenous>();

  static constexpr size_t MeasurementDim =
      OutputLayout::template total_size<OutputKind::measurement>();
  static constexpr size_t PerformanceDim =
      OutputLayout::template total_size<OutputKind::performance>();
  static constexpr size_t RegulatedDim =
      OutputLayout::template total_size<OutputKind::regulated>();
  static constexpr size_t StateOutputDim =
      OutputLayout::template total_size<OutputKind::state>();
};

/// @brief extract scalar type from plant
template <typename Plant>
using plant_type_t = typename plant_traits<linalg::remove_cvref_t<Plant>>::T;

/// @brief extract state-space type from plant
template <typename Plant>
using plant_state_space_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::StateSpace;

/// @brief extract input layout type from plant
template <typename Plant>
using plant_input_layout_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::InputLayout;

/// @brief extract output layout type from plant
template <typename Plant>
using plant_output_layout_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::OutputLayout;

/// @brief extract state vector from plant
template <typename Plant>
using plant_state_vec_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::State;

/// @brief extract full input vector from plant
template <typename Plant>
using plant_input_vec_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::Input;

/// @brief extract full output vector from plant
template <typename Plant>
using plant_output_vec_t =
    typename plant_traits<linalg::remove_cvref_t<Plant>>::Output;

/// @brief PlantModel concept
template <typename Plant>
concept PlantModelLike = requires(const Plant& plant) {
  typename plant_traits<linalg::remove_cvref_t<Plant>>::StateSpace;
  { plant.state_space };
  { plant.control_matrix() };
  { plant.disturbance_matrix() };
  { plant.measurement_matrix() };
  { plant.performance_matrix() };
};

/// @brief check if plant model is continuous
template <typename Plant>
inline constexpr bool plant_is_continuous_v =
    plant_traits<linalg::remove_cvref_t<Plant>>::DType ==
    TimeDomainType::CONTINUOUS;

/// @brief ContinuousPlantModel concept
template <typename Plant>
concept ContinuousPlantModelLike =
    PlantModelLike<Plant> && plant_is_continuous_v<Plant>;

/// @brief DiscretePlantModel concept
template <typename Plant>
concept DiscretePlantModelLike =
    PlantModelLike<Plant> && !plant_is_continuous_v<Plant>;

/// @brief Model accepted by state-feedback synthesis
template <typename Model>
concept ContinuousStateFeedbackModelLike =
    ContinuousStateSpaceLike<Model> || ContinuousPlantModelLike<Model>;

/// @brief state-feedback model traits
template <typename Model>
struct state_feedback_model_traits;

template <ContinuousStateSpaceLike SS>
struct state_feedback_model_traits<SS> {
  using Model = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<Model>;
  using State = state_vec_t<Model>;
  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<input_vec_t<Model>>;
};

template <ContinuousPlantModelLike Plant>
struct state_feedback_model_traits<Plant> {
  using Model = linalg::remove_cvref_t<Plant>;
  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;
  static constexpr size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr size_t InputDim = plant_traits<Model>::ControlDim;
};

/// @brief extract scalar type from state-feedback model
template <typename Model>
using state_feedback_type_t =
    typename state_feedback_model_traits<linalg::remove_cvref_t<Model>>::T;

/// @brief extract state vector from state-feedback model
template <typename Model>
using state_feedback_state_vec_t =
    typename state_feedback_model_traits<linalg::remove_cvref_t<Model>>::State;

/// @brief Model accepted by observer synthesis
template <typename Model>
concept ContinuousObserverModelLike =
    ContinuousStateSpaceLike<Model> || ContinuousPlantModelLike<Model>;

/// @brief observer model traits
template <typename Model>
struct observer_model_traits;

template <ContinuousStateSpaceLike SS>
struct observer_model_traits<SS> {
  using Model = linalg::remove_cvref_t<SS>;
  using T = ss_type_t<Model>;
  using State = state_vec_t<Model>;
  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t OutputDim = linalg::mat_rows_v<ss_output_t<Model>>;
};

template <ContinuousPlantModelLike Plant>
struct observer_model_traits<Plant> {
  using Model = linalg::remove_cvref_t<Plant>;
  using T = plant_type_t<Model>;
  using State = plant_state_vec_t<Model>;
  static constexpr size_t StateDim = plant_traits<Model>::StateDim;
  static constexpr size_t OutputDim = plant_traits<Model>::MeasurementDim;
};

/// @brief extract scalar type from observer model
template <typename Model>
using observer_type_t =
    typename observer_model_traits<linalg::remove_cvref_t<Model>>::T;

/// @brief extract state vector from observer model
template <typename Model>
using observer_state_vec_t =
    typename observer_model_traits<linalg::remove_cvref_t<Model>>::State;

} // namespace control
