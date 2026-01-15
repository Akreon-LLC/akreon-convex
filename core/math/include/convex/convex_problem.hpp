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

#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/**
 * @brief ConvexProblem struct
 */
template <typename Meta, BuilderLike Builder>
struct ConvexProblem {
  /**
   * @brief CTOR
   *
   * @param builder
   */
  constexpr explicit ConvexProblem(const Builder& builder_in)
      : builder(builder_in) {}

  /**
   * @brief CTOR
   *
   * @param decision var x
   * @param builder
   *
   * return new problem
   */
  constexpr explicit ConvexProblem(const Variable<Meta>& x,
                                   const Builder& builder_in)
      : builder(x.template add_var_block<AtomType::VAR>(builder_in)) {}

  /**
   * @brief add constraint
   *
   * @param atom
   *
   * @param new problem
   */
  template <AtomLike Atom>
  constexpr auto add_constraint(const Atom& atom) const {
    if constexpr (is_fn_type_v<Atom>) {
      return ConvexProblem(atom.template canonicalize<AtomType::FN>(builder));
    } else if constexpr (is_set_type_v<Atom>) {
      return ConvexProblem(atom.template canonicalize<AtomType::SET>(builder));
    } else if constexpr (is_add_type_v<Atom>) {
      return ConvexProblem(atom.template canonicalize<AtomType::ADD>(builder));
    } else if constexpr (is_cap_type_v<Atom>) {
      return ConvexProblem(atom.template canonicalize<AtomType::CAP>(builder));
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
  constexpr auto minimize(const Atom& atom) const {
    if constexpr (is_set_type_v<Atom> || is_cap_type_v<Atom>) {
      throw std::domain_error(
          "Attempting to minimize a set or intersection of sets");
    }
    static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
    if constexpr (Tag == CurvatureType::CONCAVE) {
      throw std::domain_error("Attempting to minimize a concave function");
    } else {
      if constexpr (is_fn_type_v<Atom>) {
        return ConvexProblem(atom.template canonicalize<AtomType::FN>(builder));
      } else if constexpr (is_add_type_v<Atom>) {
        return ConvexProblem(
            atom.template canonicalize<AtomType::ADD>(builder));
      } else {
        /// no-op
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
          "Attempting to minimize a set or intersection of sets");
    }
    static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      throw std::domain_error("Attempting to maximize a convex function");
    } else {
      if constexpr (is_fn_type_v<Atom>) {
        return ConvexProblem(atom.template canonicalize<AtomType::FN>(builder));
      } else if constexpr (is_add_type_v<Atom>) {
        return ConvexProblem(
            atom.template canonicalize<AtomType::ADD>(builder));
      } else {
        /// no-op
      }
    }
  }

  /** @brief member var */
  Builder builder;
};

}  // namespace convex
