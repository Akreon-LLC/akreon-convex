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
enum struct StageType : uint8_t {
  INIT = 0,
  DEC_VAR_ADDED,
  CONSTRAINTS_ADDED,
  PROBLEM_SET
};

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
inline constexpr bool is_problem_set =
    prob_traits<linalg::remove_cvref_t<Prob>>::StageV == StageType::PROBLEM_SET;

/**
 * @brief ConvexProblem struct
 */
template <BuilderLike Builder, StageType Stage = StageType::INIT>
class ConvexProblem {
 public:
  /**
   * @brief CTOR
   *
   * @param builder
   */
  constexpr explicit ConvexProblem(const Builder& builder_in)
      : builder(builder_in) {}

  /**
   * @brief add dec var
   *
   * @param variable
   *
   * @return new problem
   */
  template <typename Meta>
  constexpr auto add_dec_var(const Variable<Meta>& x) const
    requires(Stage == StageType::INIT)
  {
    const auto new_builder = x.template add_var_block<AtomType::VAR>(builder);
    return ConvexProblem<decltype(new_builder), StageType::DEC_VAR_ADDED>(
        new_builder);
  }

  /**
   * @brief in set constraint
   *
   * @param atom (set)
   *
   * @return new convex prob
   */
  template <AtomLike Atom>
  constexpr auto in_set(const Atom& atom) const
    requires(Stage == StageType::DEC_VAR_ADDED ||
             Stage == StageType::CONSTRAINTS_ADDED)
  {
    if constexpr (!is_set_type_v<Atom>)
      throw std::domain_error("Atom is not a set type");
    else {
      const auto new_builder =
          atom.template canonicalize<AtomType::SET>(builder);
      return ConvexProblem<decltype(new_builder), StageType::CONSTRAINTS_ADDED>(
          new_builder);
    }
  }

  /**
   * @brief add constraint
   *
   * TODO - update (need IN_SET, GEQ, LEQ, EQ, etc...)
   *
   * @param atom
   *
   * @param new problem
   */
#if 0
  template <AtomLike Atom>
  constexpr auto add_constraint(const Atom& atom) const
    requires(Stage == StageType::DEC_VAR_ADDED ||
             Stage == StageType::CONSTRAINTS_ADDED)
  {
    if constexpr (is_fn_type_v<Atom>) {
      const auto new_builder =
          atom.template canonicalize<AtomType::FN>(builder);
      return ConvexProblem<decltype(new_builder), StageType::CONSTRAINTS_ADDED>(
          new_builder);
    } else if constexpr (is_set_type_v<Atom>) {
    } else if constexpr (is_add_type_v<Atom>) {
      const auto new_builder =
          atom.template canonicalize<AtomType::ADD>(builder);
      return ConvexProblem<decltype(new_builder), StageType::CONSTRAINTS_ADDED>(
          new_builder);
    } else if constexpr (is_cap_type_v<Atom>) {
      const auto new_builder =
          atom.template canonicalize<AtomType::CAP>(builder);
      return ConvexProblem<decltype(new_builder), StageType::CONSTRAINTS_ADDED>(
          new_builder);
    } else {
      throw std::domain_error(
          "Attempting to add an invalid constraint to the convex problem");
    }
  }
#endif

  /**
   * @brief minimize
   *
   * @param atom
   *
   * @return new problem
   */
  template <AtomLike Atom>
  constexpr auto minimize(const Atom& atom) const
    requires(Stage == StageType::CONSTRAINTS_ADDED ||
             Stage == StageType::DEC_VAR_ADDED)
  {
    if constexpr (is_set_type_v<Atom> || is_cap_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to minimize a set or intersection of sets");
    }

    if constexpr (is_var_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to minimize over a var type, please use affine op, "
          "convex function, or index of var instead of raw type");
    }

    static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
    if constexpr (Tag == CurvatureType::CONCAVE) {
      throw std::domain_error("Attempting to minimize a concave function");
    } else {
      if constexpr (is_fn_type_v<Atom>) {
        const auto new_builder =
            atom.template canonicalize<AtomType::FN>(builder);
        return ConvexProblem<decltype(new_builder), StageType::PROBLEM_SET>(
            new_builder);
      } else if constexpr (is_add_type_v<Atom>) {
        const auto new_builder =
            atom.template canonicalize<AtomType::ADD>(builder);
        return ConvexProblem<decltype(new_builder), StageType::PROBLEM_SET>(
            new_builder);
      } else {
        const auto new_builder =
            atom.template canonicalize<AtomType::VAR>(builder);
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
  constexpr auto maximize(const Atom& atom) const
    requires(Stage == StageType::CONSTRAINTS_ADDED ||
             Stage == StageType::DEC_VAR_ADDED)
  {
    if constexpr (is_set_type_v<Atom> || is_cap_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to maximize a set or intersection of sets");
    }

    if constexpr (is_var_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to maximize over a var type, please use affine op, "
          "concave function, or index of var instead of raw type");
    }

    static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      throw std::domain_error("Attempting to maximize a convex function");
    } else {
      if constexpr (is_fn_type_v<Atom>) {
        return minimize(atom.template mul<-1.0>());
      } else if constexpr (is_add_type_v<Atom>) {
        /// TODO - add mul -1 to sum map here
      } else {
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
  /** @brief member var */
  Builder builder;
};

}  // namespace convex
