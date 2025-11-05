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
 * min  f_0(x)
 * s.t. f_i(x) <= 0
 *      A x = b
 *
 * where f_i are convex and A x = b is an affine equality
 */
template <linalg::RnSpace Domain, typename... Constraints>
class ConvexProgram {
 public:
  /// @brief alias to extract type traints from domain
  using T = linalg::space_type_t<Domain>;
  using Point = linalg::space_point_t<Domain>;

  /// @brief helper to get dimension of domain
  static constexpr size_t N = linalg::vec_size_v<Point>;
  static constexpr size_t M = sizeof...(Constraints);

  /**
   * @brief CTOR
   *
   * Objective function casted to epigraph convex problem -> cone program
   *
   * min t
   * s.t. f_0(x) - t <= 0
   *      f_i(x) <= 0
   *      A x = b
   *
   * @param parameter pack of functions
   */
  constexpr explicit ConvexProgram(Constraints... cs)
      : constraints(std::forward<Constraints>(cs)...) {
    (perform_checks(cs), ...);
  }

  /**
   * @brief solve convex program
   *
   * @return T/F
   */
  constexpr bool solve() const {
    const auto cones = transform_constraints_to_cones();
    auto make_cone_prog = [](const auto&... args) {
      return ConeProgram<Domain, decltype(args)...>(
          std::forward<decltype(args)>(args)...);
    };
    const auto cone_prog = std::apply(make_cone_prog, cones);
    return cone_prog.soft_check() ? cone_prog.solve() : false;
  }

 private:
  /**
   * @brief transform convex constraints to cones
   *
   * (1) Convex Fn -> epigraph -> cone view
   * (2) Concave Fn -> negate to convex fn -> epigraph -> cone view
   * (3) Affine Equality -> cone lift (view)
   * (4) TODO handle nested functions
   *
   * @return tuple of cones
   */
  constexpr auto transform_constraints_to_cones() const {
    auto to_epi = [&](const auto& arg) {
      if constexpr (AffineSetLike<decltype(arg)>)
        return arg.cone_view();
      else if constexpr (ConvexFnLike<decltype(arg)>)
        return arg.epigraph();
      else if constexpr (ConcaveFnLike<decltype(arg)>)
        return arg.negate().epigraph();
      else
        static_assert(false, "This should never get hit!");
    };
    return std::apply(
        [&](const auto&... cs) {
          return std::make_tuple(to_epi(std::forward<decltype(cs)>(cs))...);
        },
        constraints);
  }

  /**
   * @brief perform checks on incoming functions
   *
   * static evaluation checks (compilation error if one fails)
   *
   * @return T/F
   */
  constexpr void perform_checks(const auto& constraint) const {
    static_assert(ConvexFnLike<decltype(constraint)> ||
                      ConcaveFnLike<decltype(constraint)> ||
                      AffineSetLike<decltype(constraint)>,
                  "Constraint function does not meet convex, concave, or "
                  "affine criteria");
    if constexpr (AffineSetLike<decltype(constraint)>) {
      using ConstraintDom = set_point_t<decltype(constraint)>;
      static_assert(std::same_as<ConstraintDom, set_point_t<Domain>>,
                    "Constraint domain is mismatched with problem domain");
    } else {
      using ConstraintDom = convex_fn_domain_t<decltype(constraint)>;
      static_assert(std::same_as<ConstraintDom, Domain>,
                    "Constraint domain is mismatched with problem domain");
    }
  }

  /** @brief member vars for convex program */
  std::tuple<Constraints...>
      constraints;  //< tuple of constraints (convex + affine)
};

}  // namespace convex
