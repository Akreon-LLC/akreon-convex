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

#include "cone_program.hpp"
#include "sets/affine_subspace.hpp"

namespace convex {

/**
 * @brief Convex Program formulation
 *
 * min  f_(x)
 * s.t. f_i(x) <= 0
 *      A x = b
 *
 * where f_i are convex and A x = b is an affine equality
 */
template <linalg::RnSpace Domain, typename... Fns>
class ConvexProgram {
 public:
  /// @brief alias to extract type traints from domain
  using T = linalg::space_type_t<Domain>;
  using Point = linalg::space_point_t<Domain>;

  /// @brief helper to get dimension of domain
  static constexpr size_t N = linalg::vec_size_v<Point>;
  static constexpr size_t M = sizeof...(Fns);

  /**
   * @brief CTOR
   *
   * @param parameter pack of functions
   */
  constexpr explicit ConvexProgram(Fns... fns)
      : funcs(std::forward<Fns>(fns)...) {
    (perform_checks(fns), ...);
  }

  /**
   * @brief solve convex program
   *
   * @return T/F
   */
  constexpr bool solve() const {
    const auto cones = transform_funcs_to_cones();
    auto make_cone_prog = [](const auto&... args) {
      return ConeProgram<Domain, decltype(args)...>(
          std::forward<decltype(args)>(args)...);
    };
    const auto cone_prog = std::apply(make_cone_prog, cones);
    return cone_prog.soft_check() ? cone_prog.solve() : false;
  }

 private:
  /**
   * @brief transform functions to cones
   *
   * (1) Convex Fn -> epigraph -> cone view
   * (2) Concave Fn -> negate to convex fn -> epigraph -> cone view
   * (3) TODO nested functions
   *
   * @return tuple of cones
   */
  constexpr auto transform_funcs_to_cones() const {
    auto to_epi = [&](const auto& fn) {
      if constexpr (ConvexFnLike<decltype(fn)>)
        return fn.epigraph();
      else
        return fn.negate().epigraph();
    };
    return std::apply(
        [&](const auto&... fns) {
          return std::make_tuple(to_epi(std::forward<decltype(fns)>(fns))...);
        },
        funcs);
  }

  /**
   * @brief perform checks on incoming functions
   *
   * static evaluation checks (compilation error if one fails)
   *
   * @return T/F
   */
  constexpr void perform_checks(const auto& fn) const {
    using FnDom = convex_fn_domain_t<decltype(fn)>;
    using FnT = linalg::space_type_t<FnDom>;
    static_assert(std::same_as<FnT, T>,
                  "Function domain type is mismatched with problem domain");
    static_assert(N == linalg::space_dim_v<FnDom>,
                  "Function domain dim is mismatched with problem dim");
    static_assert(ConvexFnLike<decltype(fn)> || ConcaveFnLike<decltype(fn)>,
                  "Provided function does not meet convex or concave criteria");
  }

  /** @brief member vars for convex program */
  std::tuple<Fns...> funcs;  //< tuple of functions
};

}  // namespace convex
