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
      throw std::domain_error("Atom is not a set type");
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
    (void)lhs, (void)rhs;
    static constexpr bool lhs_valid = is_fn_type_v<LHS> || is_index_type_v<LHS>;
    if constexpr (!lhs_valid) {
      throw std::domain_error("LHS must be a valid atom (fn or index of var)");
    }

    static constexpr bool rhs_valid =
        AtomLike<RHS> ? (is_fn_type_v<RHS> || is_index_type_v<RHS>)
                      : linalg::RealLike<RHS>;
    if constexpr (!rhs_valid) {
      throw std::domain_error(
          "RHS must be either a valid atom (fn or index of var) or a scalar "
          "type");
    }

    /// TODO form nonneg orthant cone 0 <= rhs - lhs and create new prob
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
    const auto set_rhs = [&](const auto& new_lhs) {
      if constexpr (is_var_type_v<RHS>) {
        const auto new_rhs = rhs.mul(-1.0);
        return leq<LHS, decltype(new_rhs)>(new_lhs, new_rhs);
      } else {
        const auto new_rhs = rhs.template mul<-1.0>();
        return leq<LHS, decltype(new_rhs)>(new_lhs, new_rhs);
      }
    };
    if constexpr (is_var_type_v<LHS>) {
      const auto new_lhs = lhs.mul(-1.0);
      if constexpr (AtomLike<RHS>)
        return set_rhs(new_lhs);
      else if constexpr (linalg::RealLike<RHS>)
        return leq<LHS, RHS>(new_lhs, T(-1) * rhs);
      else
        throw std::domain_error(
            "Invalid RHS type (must be atom like or scalar)");
    } else if constexpr (is_fn_type_v<LHS>) {
      const auto new_lhs = lhs.template mul<-1.0>();
      if constexpr (AtomLike<RHS>)
        return set_rhs(new_lhs);
      else if constexpr (linalg::RealLike<RHS>)
        return leq<LHS, RHS>(new_lhs, T(-1) * rhs);
      else
        throw std::domain_error(
            "Invalid RHS type (must be atom like or scalar)");
    }
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
      throw std::domain_error(
          "Attempting to minimize a set or intersection of sets");
    } else if constexpr (is_var_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to minimize over a var type, please use affine op, "
          "convex function, or index of var instead of raw type");
    } else {
      static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        throw std::domain_error("Attempting to minimize a concave function");
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
      throw std::domain_error(
          "Attempting to maximize a set or intersection of sets");
    } else if constexpr (is_var_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to maximize over a var type, please use affine op, "
          "concave function, or index of var instead of raw type");
    } else {
      static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
      if constexpr (Tag == CurvatureType::CONVEX) {
        throw std::domain_error("Attempting to maximize a convex function");
      } else {
        if constexpr (is_fn_type_v<Atom> || is_add_type_v<Atom>) {
          return minimize(atom.template mul<-1.0>());
        } else {
          return minimize(atom.mul(-1.0));
        }
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
  /** @brief member var */
  Builder builder;
};

}  // namespace convex
