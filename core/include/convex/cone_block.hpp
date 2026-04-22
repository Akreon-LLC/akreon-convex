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

#include "cone_traits.hpp"

namespace conic {

/**
 * @brief Cone block
 *
 * Houses the convex object -> cone block structure
 */
template <affine::AffineMapLike Map, typename... Cs>
struct ConeBlock {
  /// @brief alias to extract scalar type from space
  using MatT = affine::affine_mat_t<Map>;
  using T = linalg::mat_type_t<MatT>;

  /// @brief alias to extract codom & dom of affine map
  static constexpr size_t M = linalg::mat_rows_v<MatT>;
  static constexpr size_t N = linalg::mat_cols_v<MatT>;

  /**
   * @brief CTOR for cone block
   *
   * @param affine map (takes convex object -> cone coordinates)
   * @param param pack of cones
   */
  constexpr explicit ConeBlock(const Map& map_in, Cs... cs)
      : map(map_in), cones(std::forward<Cs>(cs)...) {}

  /**
   * @brief CTOR for cone block
   *
   * @param affine map
   * @param tuple of cones
   */
  constexpr explicit ConeBlock(const Map& map_in, std::tuple<Cs...> cs)
      : map(map_in), cones(std::move(cs)) {}

  /**
   * @brief validate cone block
   *
   * @return T/F
   */
  constexpr bool validate() const {
    // (dakre) lambda functor to perform the soft check
    size_t n_dims = 0;
    auto perform_check = [&](const auto& arg) {
      using ArgT = cone_scalar_t<decltype(arg)>;
      static_assert(
          std::same_as<ArgT, T>,
          "Cone scalar type is mismatched with cone block scalar type");
      n_dims += cone_dim_t<decltype(arg)>;
    };
    // (dakre) apply fold expression over comma operator on stored cones
    std::apply([&](const auto&... args) { (perform_check(args), ...); }, cones);
    return n_dims == M;
  }

  /** @brief member vars for cone block */
  Map map; ///< affine map taking the decision variables -> cone coords
  std::tuple<Cs...> cones; ///< tuple of cones aligned with map
};

} // namespace conic
