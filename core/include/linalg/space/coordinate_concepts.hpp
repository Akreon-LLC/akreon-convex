/* Copyright © 2025 Akreon LLC. All rights reserved. */

#pragma once

#include <concepts>

#include "coordinate_map.hpp"

namespace linalg {

template <class Candidate>
concept Space = requires {
  typename space_scalar_t<Candidate>;
  typename space_element_t<Candidate>;
  { space_dim_v<Candidate> } -> std::convertible_to<std::size_t>;
};

template <class Candidate>
concept CoordinatizedSpace =
    Space<Candidate> &&
    requires(const space_element_t<Candidate>& element,
             const space_coordinate_t<Candidate>& coordinate) {
      {
        coordinates<Candidate>(element)
      } -> std::same_as<space_coordinate_t<Candidate>>;
      {
        from_coordinates<Candidate>(coordinate)
      } -> std::same_as<space_element_t<Candidate>>;
    };

} // namespace linalg
