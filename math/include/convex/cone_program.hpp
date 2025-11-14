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

namespace convex {

/**
 * @brief Cone Program formulation
 *
 * min. c^T x
 * s.t. F x + b <=_k 0
 *
 * AKA
 * min. c^T x
 * s.t. [s_0 s_1 ... s_m]^T = [A F_1 F_2 ... F_{m-1}]^T x + [-b g_1 g_2 ...
 * g_{m-1}]^T
 *
 * where s vector \in K_0 x K_1 ... K_m cartesian product of cones
 *
 * so equivalent problem becomes the following:
 * min. c^T z
 * s.t. F z + b \in K
 */
template <linalg::RnSpace Domain, ConeViewLike... CVs>
class ConeProgram {
 public:
  /// @brief alias to extract type traints from domain
  using T = linalg::space_type_t<Domain>;
  using Point = linalg::space_point_t<Domain>;

  /// @brief helper to get dimension of domain
  static constexpr size_t N = linalg::vec_size_v<Point>;
  static constexpr size_t M = sizeof...(CVs);

  /**
   * @brief CTOR
   *
   * @param parameter pack of cone views
   */
  constexpr explicit ConeProgram(CVs... cvs)
      : cone_blocks(std::forward<CVs>(cvs)...) {
    // (dakre) by default fill last N:N+M elements to 1 (i.e., linear objective
    // on slack vars)
    for (size_t k = N; k < N + M; ++k) c[k] = T(1);
  }

  /**
   * @brief soft check
   *
   * Validation step before feasibility -> solver runs to see if the problem is
   * formed correctly.
   *
   * There are scenarios where we won't know if the cone is pointed or solid
   * until running feasibility, so the tri-state return will be false on false
   * cone check and passes otherwise
   */
  constexpr bool soft_check() const {
    bool rv = true;
    // (dakre) lambda functor to perform the soft check
    auto perform_check = [&](const auto& arg) {
      using ArgT = cone_view_type_t<decltype(arg)>;
      static_assert(std::same_as<ArgT, T>,
                    "Cone view type is mismatched with domain vector in "
                    "program formulation");
      static_assert(N == cone_view_dim_v<decltype(arg)>,
                    "Cone view dim is mismatched with domain vector in program "
                    "formulation");
      rv &= arg.is_proper() != ConeCheck::False;
    };
    // (dakre) apply fold expression over comma operator on stored tuple of cone
    // blocks
    std::apply([&](const auto&... args) { (perform_check(args), ...); },
               cone_blocks);
    return rv;
  }

  /**
   * @brief solve
   *
   * @return SolverResult enum
   */
  constexpr auto solve() const { return linalg::SolverResult::OPTIMAL; }

 private:
  /** @brief member vars for cone program */
  std::tuple<CVs...> cone_blocks;  //< K = k_0 x k_1 ... x k_{m} cones
  linalg::Vector<T, N + M> c;      //< linear weighting for objective fn
};

}  // namespace convex
