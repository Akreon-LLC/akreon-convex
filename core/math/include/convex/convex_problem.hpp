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

#include "convex/convex_util.hpp"

namespace convex {

/// @brief StageType for forming convex problem
enum struct StageType : uint8_t { INIT = 0, IN_PROGRESS, PROBLEM_SET };

/// @brief forward declaration of convex problem
template <BuilderLike Builder, StageType Stage>
class ConvexProblem;

template <typename Prob>
struct prob_traits;

template <BuilderLike Builder, StageType Stage>
struct prob_traits<ConvexProblem<Builder, Stage>> {
  using BuilderT = Builder;
  static constexpr StageType StageV = Stage;
};

/// @brief builder type from prob
template <typename Prob>
using prob_builder_t =
    typename prob_traits<linalg::remove_cvref_t<Prob>>::BuilderT;

/// @brief stage type from prob
template <typename Prob>
inline constexpr bool is_problem_set_v =
    prob_traits<linalg::remove_cvref_t<Prob>>::StageV == StageType::PROBLEM_SET;

/**
 * @brief make problem
 *
 * @param dec var x
 *
 * @return convex problem
 */
template <AtomLike Var>
constexpr auto make_problem(const Var& x) {
  using Space = dom_space_t<meta_dom_t<atom_meta_t<Var>>>;
  const auto builder = create_empty_cblk_builder<Space>();
  const auto new_builder = x.template add_var_block<AtomType::VAR>(builder);
  return ConvexProblem<decltype(new_builder), StageType::INIT>(new_builder);
}

/**
 * @brief ConvexProblem struct
 */
template <BuilderLike Builder, StageType Stage>
class ConvexProblem {
 public:
  /// @brief alias for scalar type
  using T = linalg::space_type_t<builder_dom_t<Builder>>;

  /**
   * @brief CTOR
   *
   * @param builder
   */
  constexpr explicit ConvexProblem(const Builder& builder_in)
      : builder(builder_in) {}

  /**
   * @brief in set constraint
   *
   * (ex) A x = b is an affine subspace in set constraint
   *
   * @param atom (set)
   *
   * @return new convex prob
   */
  template <AtomLike Atom>
  constexpr auto in_set(const Atom& atom) const {
    if constexpr (!(is_set_type_v<Atom> || is_cap_type_v<Atom>))
      static_assert(false, "Atom is not a set type");
    else {
      const auto new_builder =
          atom.template canonicalize<AtomType::CAP>(builder);
      return ConvexProblem<decltype(new_builder), StageType::IN_PROGRESS>(
          new_builder);
    }
  }

  /**
   * @brief leq
   *
   * lhs <= rhs... 0 <= rhs - lhs
   *
   * @param lhs atom
   * @param rhs atom
   *
   * @return new convex problem
   */
  template <AtomLike LHS, typename RHS>
  constexpr auto leq(const LHS& lhs, const RHS& rhs) const {
    check_ineq_types<LHS, RHS>();

    const auto new_lhs = negate_lhs(lhs);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    const auto cones = std::make_tuple(rplus);

    if constexpr (AtomLike<RHS>) {
      using Dom = linalg::Matrix<T, 1, 2>;
      using Codom = linalg::Matrix<T, 1, 1>;
      const Sum<SumMeta<Dom, Codom>, decltype(new_lhs), RHS> sum(new_lhs, rhs);
      const auto new_builder =
          sum.template canonicalize<AtomType::ADD>(builder);
      const auto builder_up = new_builder.template make<AtomType::INEQ>(cones);
      return ConvexProblem<decltype(builder_up), StageType::IN_PROGRESS>(
          builder_up);
    } else {
      const auto A = new_lhs.lift_map.linear_part();
      const auto b = new_lhs.lift_map.translation() + linalg::Vector<T, 1>{rhs};
      const auto new_builder = builder.template make<AtomType::INEQ>(
          -1, affine::AffineMap<decltype(A)>(A, b), cones);
      return ConvexProblem<decltype(new_builder), StageType::IN_PROGRESS>(
          new_builder);
    }
  }

  /**
   * @brief geq
   *
   * lhs >= rhs... lhs - rhs >= 0
   * -lhs <= -rhs
   *
   * @param lhs atom
   * @param rhs atom
   *
   * @return new convex problem
   */
  template <AtomLike LHS, typename RHS>
  constexpr auto geq(const LHS& lhs, const RHS& rhs) const {
    check_ineq_types<LHS, RHS>();

    const auto negate_rhs = [&]() {
      if constexpr (is_index_type_v<RHS>) {
        return rhs.mul(-1.0);
      } else if constexpr (AtomLike<RHS>) {
        return rhs.template mul<-1.0>();
      } else {
        return T(-1) * rhs;
      }
    };

    const auto new_lhs = negate_lhs(lhs);
    const auto new_rhs = negate_rhs();
    return leq<decltype(new_lhs), decltype(new_rhs)>(new_lhs, new_rhs);
  }

  /**
   * @brief minimize
   *
   * @param atom
   *
   * @return new problem
   */
  template <AtomLike Atom>
  constexpr auto minimize(const Atom& atom) const {
    if constexpr (is_set_type_v<Atom> || is_cap_type_v<Atom>) {
      static_assert(false,
                    "Attempting to minimize a set or intersection of sets");
    } else if constexpr (is_var_type_v<Atom>) {
      static_assert(
          false,
          "Attempting to minimize over a var type, please use affine op, "
          "convex function, or index of var instead of raw type");
    } else {
      static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        static_assert(false, "Attempting to minimize a concave function");
      } else {
        const auto new_builder =
            atom.template canonicalize<AtomType::CAP>(builder);
        return ConvexProblem<decltype(new_builder), StageType::PROBLEM_SET>(
            new_builder);
      }
    }
  }

  /**
   * @brief maximize
   *
   * @param atom
   *
   * @return new problem
   */
  template <AtomLike Atom>
  constexpr auto maximize(const Atom& atom) const {
    if constexpr (is_set_type_v<Atom> || is_cap_type_v<Atom>) {
      static_assert(false,
                    "Attempting to maximize a set or intersection of sets");
    } else if constexpr (is_var_type_v<Atom>) {
      static_assert(
          false,
          "Attempting to maximize over a var type, please use affine op, "
          "concave function, or index of var instead of raw type");
    } else {
      static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
      if constexpr (Tag == CurvatureType::CONVEX) {
        static_assert(false, "Attempting to maximize a convex function");
      } else {
        if constexpr (is_fn_type_v<Atom> || is_add_type_v<Atom>)
          return minimize(atom.template mul<-1.0>());
        else
          return minimize(atom.mul(-1.0));
      }
    }
  }

  /**
   * @brief validate
   *
   * @return bool
   */
  constexpr bool validate() const { return builder.cb.validate(); }

  /**
   * @brief linear objective
   *
   * min c^T z
   *
   * @return c
   */
  constexpr auto linear_objective() const
    requires(Stage == StageType::PROBLEM_SET)
  {
    return builder.coord_map.linear_part();
  }

  /**
   * @brief constraint map (canonical form) [ F; g ]
   *
   * F z + g
   *
   * @return F
   */
  constexpr auto constraint_map() const
    requires(Stage == StageType::PROBLEM_SET)
  {
    return builder.cb.map;
  }

  /**
   * @brief cones
   *
   * @return tuple of cones
   */
  constexpr auto constraint_cones() const
    requires(Stage == StageType::PROBLEM_SET)
  {
    return builder.cb.cones;
  }

 private:
  /**
   * @brief check inequality types
   *
   * @return void
   */
  template <AtomLike LHS, typename RHS>
  constexpr void check_ineq_types() const {
    static constexpr bool lhs_valid = is_fn_type_v<LHS> || is_index_type_v<LHS>;
    if constexpr (!lhs_valid) {
      static_assert(false, "LHS must be a valid atom (fn or index of var)");
    }

    static constexpr bool rhs_valid =
        AtomLike<RHS> ? (is_fn_type_v<RHS> || is_index_type_v<RHS>)
                      : linalg::RealLike<RHS>;
    if constexpr (!rhs_valid) {
      static_assert(
          false,
          "RHS must be either a valid atom (fn or index of var) or a scalar "
          "type");
    }
  }

  /**
   * @brief negate lhs
   *
   * @return negated LHS
   */
  template <AtomLike LHS>
  constexpr auto negate_lhs(const LHS& lhs) const {
    if constexpr (is_index_type_v<LHS>)
      return lhs.mul(-1.0);
    else
      return lhs.template mul<-1.0>();
  }

  /** @brief member var */
  Builder builder;
};

}  // namespace convex
