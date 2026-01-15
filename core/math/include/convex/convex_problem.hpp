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

/**
 * @brief ConvexProblem struct
 */
template <BuilderLike Builder, bool DecVarAdded = false,
          bool ConstraintsAdded = false, bool ProblemSet = false>
struct ConvexProblem {
  /**
   * @brief CTOR
   */
  constexpr explicit ConvexProblem()
    requires(!DecVarAdded)
      : builder(create_empty_cblk_builder<builder_dom_t<Builder>>()) {}

  /**
   * @brief add dec var
   *
   * @param variable
   *
   * @return new problem
   */
  template <typename Meta>
  constexpr auto add_dec_var(const Variable<Meta>& x) const
    requires(!DecVarAdded)
  {
    const auto new_builder = x.template add_var_block<AtomType::VAR>(builder);
    return ConvexProblem<decltype(new_builder), true>(new_builder);
  }

  /**
   * @brief CTOR
   *
   * @param builder
   */
  constexpr explicit ConvexProblem(const Builder& builder_in)
    requires(DecVarAdded && !ProblemSet)
      : builder(builder_in) {}

  /**
   * @brief add constraint
   *
   * @param atom
   *
   * @param new problem
   */
  template <AtomLike Atom>
  constexpr auto add_constraint(const Atom& atom) const
    requires(DecVarAdded && !ProblemSet)
  {
    if constexpr (is_fn_type_v<Atom>) {
      const auto new_builder =
          atom.template canonicalize<AtomType::FN>(builder);
      return ConvexProblem<decltype(new_builder), DecVarAdded>(new_builder);
    } else if constexpr (is_set_type_v<Atom>) {
      const auto new_builder =
          atom.template canonicalize<AtomType::SET>(builder);
      return ConvexProblem<decltype(new_builder), DecVarAdded>(new_builder);
    } else if constexpr (is_add_type_v<Atom>) {
      const auto new_builder =
          atom.template canonicalize<AtomType::ADD>(builder);
      return ConvexProblem<decltype(new_builder), DecVarAdded>(new_builder);
    } else if constexpr (is_cap_type_v<Atom>) {
      const auto new_builder =
          atom.template canonicalize<AtomType::CAP>(builder);
      return ConvexProblem<decltype(new_builder), DecVarAdded>(new_builder);
    } else {
      throw std::domain_error(
          "Attempting to add an invalid constraint to the convex problem");
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
  constexpr auto minimize(const Atom& atom) const
    requires(DecVarAdded && ConstraintsAdded)
  {
    if constexpr (is_set_type_v<Atom> || is_cap_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to minimize a set or intersection of sets");
    }
    static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
    if constexpr (Tag == CurvatureType::CONCAVE) {
      throw std::domain_error("Attempting to minimize a concave function");
    } else {
      if constexpr (is_fn_type_v<Atom>) {
        const auto new_builder =
            atom.template canonicalize<AtomType::FN>(builder);
        return ConvexProblem<decltype(new_builder), DecVarAdded,
                             ConstraintsAdded, true>(new_builder);
      } else if constexpr (is_add_type_v<Atom>) {
        const auto new_builder =
            atom.template canonicalize<AtomType::ADD>(builder);
        return ConvexProblem<decltype(new_builder), DecVarAdded,
                             ConstraintsAdded, true>(new_builder);
      } else {
        const auto new_builder =
            atom.template canonicalize<AtomType::VAR>(builder);
        return ConvexProblem<decltype(new_builder), DecVarAdded,
                             ConstraintsAdded, true>(new_builder);
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
    requires(DecVarAdded && ConstraintsAdded)
  {
    if constexpr (is_set_type_v<Atom> || is_cap_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to maximize a set or intersection of sets");
    }
    static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      throw std::domain_error("Attempting to maximize a concave function");
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

  /** @brief member var */
  Builder builder;
};

}  // namespace convex
