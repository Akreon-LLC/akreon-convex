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

#include "canonical_form_builder.hpp"

namespace convex {

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
  const auto builder = conic::create_empty_cblk_builder<Space>();
  const auto new_builder = x.template add_var_block<AtomType::VAR>(builder);
  return ConvexProblem<decltype(new_builder), StageType::INIT>(new_builder);
}

/**
 * @brief ConvexProblem
 *
 * min  c^T z
 * s.t. F z + g \in K
 *
 * transformed to a fully cone problem
 * min c^T x
 * s.t. x = F z + g
 *      x \in K = K1 x K2 ... Kn
 *
 * The objective c pushed forward through linear functional c_x when x is
 * feasible: c^T z = <c_x, x> s.t. { x | \exists z, F z + g = x }
 */
template <conic::BuilderLike Builder, StageType Stage>
struct ConvexProblem {
  /// @brief alias for scalar type
  using T = linalg::space_type_t<conic::builder_dom_t<Builder>>;

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
          atom.template canonicalize<AtomType::SET>(builder);
      return ConvexProblem<decltype(new_builder), StageType::IN_PROGRESS>(
          new_builder);
    }
  }

  /**
   * @brief add ineq constraint
   *
   * @param ineq (Inequality)
   *
   * @return new convex prob
   */
  template <InequalityType IType, AtomLike LHS, typename RHS>
  constexpr auto add_constraint(const Inequality<IType, LHS, RHS>& ineq) const {
    const auto new_builder = ineq.template canonicalize<AtomType::CAP>(builder);
    return ConvexProblem<decltype(new_builder), StageType::IN_PROGRESS>(
        new_builder);
  }

  /**
   * @brief feasibility
   *
   * @return problem set
   */
  constexpr auto feasibility() const {
    const auto cb = builder.cb;
    static constexpr size_t ConeN =
        linalg::mat_cols_v<decltype(cb.map.linear_part())>;
    static constexpr size_t EqN =
        linalg::mat_cols_v<decltype(builder.eq_map.linear_part())>;
    static constexpr size_t AugN = (ConeN > EqN) ? ConeN : EqN;
    linalg::Matrix<T, 1, AugN> c{};
    affine::AffineMap<decltype(c)> new_map(c);

    const auto new_builder =
        conic::CanonicalFormBuilder<conic::builder_dom_t<Builder>,
                                    decltype(new_map), decltype(cb),
                                    decltype(builder.eq_map)>(new_map, cb,
                                                              builder.eq_map);
    return ConvexProblem<decltype(new_builder), StageType::PROBLEM_SET>(
        new_builder);
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
    } else if constexpr (is_index_type_v<Atom>) {
      const auto cb = builder.cb;
      static constexpr size_t ConeN =
          linalg::mat_cols_v<decltype(cb.map.linear_part())>;
      static constexpr size_t EqN =
          linalg::mat_cols_v<decltype(builder.eq_map.linear_part())>;
      static constexpr size_t AugN = (ConeN > EqN) ? ConeN : EqN;
      linalg::Matrix<T, 1, AugN> c{};
      auto b = builder.coord_map.translation()(0) + atom.map.translation()(0);
      static constexpr size_t N =
          linalg::space_dim_v<conic::builder_dom_t<Builder>>;
      constexpr size_t CopyN = (N < AugN) ? N : AugN;
      for (size_t i = 0; i < CopyN; ++i) c(0)(i) = atom.map.linear_part()(0)(i);
      const affine::AffineMap<decltype(c)> new_map(c, b);
      const auto new_builder =
          conic::CanonicalFormBuilder<conic::builder_dom_t<Builder>,
                                      decltype(new_map), decltype(cb),
                                      decltype(builder.eq_map)>(new_map, cb,
                                                                builder.eq_map);
      return ConvexProblem<decltype(new_builder), StageType::PROBLEM_SET>(
          new_builder);
    } else {
      static constexpr CurvatureType Tag = meta_curv_v<atom_meta_t<Atom>>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        static_assert(false, "Attempting to minimize a concave function");
      } else {
        const auto obj_builder =
            atom.template canonicalize<AtomType::CAP>(builder);
        static constexpr size_t ConeN =
            linalg::mat_cols_v<decltype(obj_builder.cb.map.linear_part())>;
        static constexpr size_t EqN =
            linalg::mat_cols_v<decltype(obj_builder.eq_map.linear_part())>;
        static constexpr size_t FullN = (ConeN > EqN) ? ConeN : EqN;
        static constexpr size_t ObjN =
            linalg::mat_cols_v<decltype(obj_builder.coord_map.linear_part())>;

        linalg::Matrix<T, 1, FullN> c{};
        constexpr size_t CopyN = (ObjN < FullN) ? ObjN : FullN;
        for (size_t i = 0; i < CopyN; ++i)
          c(0)(i) = obj_builder.coord_map.linear_part()(0)(i);

        const auto b = obj_builder.coord_map.translation()(0);
        const auto obj_map = affine::AffineMap<decltype(c)>(c, b);
        const auto new_builder = conic::CanonicalFormBuilder<
            conic::builder_dom_t<decltype(obj_builder)>, decltype(obj_map),
            decltype(obj_builder.cb), decltype(obj_builder.eq_map)>(
            obj_map, obj_builder.cb, obj_builder.eq_map);
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
   * @brief constraint map (canonical conic form) [ F; g ]
   *
   * F z + g
   *
   * @return conic constraint map
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

  /**
   * @brief affine equality map [ A; b ]
   *
   * A z = b
   *
   * @return affine equality map
   */
  constexpr auto equality_map() const
    requires(Stage == StageType::PROBLEM_SET)
  {
    return builder.eq_map;
  }

  /** @brief member var */
  Builder builder;
};

}  // namespace convex
