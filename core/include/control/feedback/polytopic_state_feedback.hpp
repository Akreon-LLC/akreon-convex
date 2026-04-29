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

#include "control/feedback/state_feedback.hpp"
#include "convex/composed/quad_form.hpp"
#include "convex/composed/weighted_sum.hpp"
#include "convex/convex_problem.hpp"
#include "convex/intersection.hpp"
#include "convex/variable.hpp"

namespace control {

/**
 * @brief PolytopicStateFeedback
 *
 * Polytopic State Feedback (PLDI)
 * Given the LTV system
 *   xdot = A(t) x + B_u(t) u
 *
 * We seek a static state feedback controller
 *   u = K x
 *
 * s.t. the CL system is quadratically stable.
 *
 * There exists Q > 0, Q = P^{-1} and Y = K Q so
 *   Q A_i^T + A_i Q + B_{ui} Y + Y^T B_{ui}^T < 0
 *
 * This set is represented over the merged decision variable
 *   z = [ svec(Q); vec(Y) ].
 */
template <ContinuousStateSpaceLike SS, size_t N>
struct PolytopicStateFeedback {
  using StateSpace = linalg::remove_cvref_t<SS>;

  using T = ss_type_t<StateSpace>;
  using VertexFeedback = StateFeedback<StateSpace>;

  static constexpr size_t StateDim = VertexFeedback::StateDim;
  static constexpr size_t InputDim = VertexFeedback::InputDim;
  static constexpr size_t QDim = VertexFeedback::QDim;
  static constexpr size_t YDim = VertexFeedback::YDim;
  static constexpr size_t ZDim = VertexFeedback::ZDim;

  using QMat = typename VertexFeedback::QMat;
  using YMat = typename VertexFeedback::YMat;
  using KMat = typename VertexFeedback::KMat;
  using FMat = typename VertexFeedback::FMat;
  using ZSpace = typename VertexFeedback::ZSpace;
  using Fs = typename VertexFeedback::Fs;
  using Polytope = linalg::Vector<SS, N>;

  /**
   * @brief operator overload (form polytopic state feedback)
   *
   * @param poly (Polytope)
   * @param Z (dec var [svec(Q); vec(Y)])
   *
   * @return PLDI as intersection of LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto operator()(const Polytope& poly, const ZVar& Z) const {
    return apply(poly, Z);
  }

  /**
   * @brief apply (polytopic state feedback)
   *
   * @param poly (Polytope)
   * @param Z (dec var [svec(Q); vec(Y)])
   *
   * @return PLDI as intersection of LMI
   */
  template <convex::AtomLike ZVar>
  constexpr auto apply(const Polytope& poly, const ZVar& Z) const {
    static_assert(convex::is_var_type_v<ZVar>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<ZVar>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(Q); vec(Y)]");

    auto make_vertex_lmi = [&]<size_t idx>() {
      return VertexFeedback()(poly(idx), Z);
    };

    auto form_intersection = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return convex::Intersection(make_vertex_lmi.template operator()<Is>()...);
    };

    return form_intersection(std::make_index_sequence<N>{});
  }

  /**
   * @brief K
   *
   * K = Y Q^{-1}
   *
   * @param Q (symmetric Lyapunov variable)
   * @param Y (lifted feedback product Y = KQ)
   *
   * @return K
   */
  constexpr auto K(const QMat& Q, const YMat& Y) const {
    return VertexFeedback().K(Q, Y);
  }
};

/**
 * @brief make PLDI Q >= margin I LMI over merged z = [svec(Q); vec(Y)]
 *
 * @param Z (merged decision var)
 * @param margin (PSD margin)
 *
 * @return LMI selecting Q-block and constraining it to PSD
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_q_constraint(
    const ZVar& Z, ss_type_t<SS> margin = 1e-6) {
  return make_state_feedback_q_constraint<SS>(Z, margin);
}

/**
 * @brief make linear objective over merged PLDI variable
 *
 * obj(z) = q_w^T svec(Q) + y_w^T vec(Y)
 *
 * @param q_weights weights over svec(Q)
 * @param y_weights weights over vec(Y)
 * @param Z merged decision var
 *
 * @return linear objective atom
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_linear_objective(
    const linalg::Vector<ss_type_t<SS>, PolytopicStateFeedback<SS, N>::QDim>&
        q_weights,
    const linalg::Vector<ss_type_t<SS>, PolytopicStateFeedback<SS, N>::YDim>&
        y_weights,
    const ZVar& Z) {
  using T = ss_type_t<SS>;
  using PSF = PolytopicStateFeedback<SS, N>;

  linalg::Vector<T, PSF::ZDim> c{};
  for (size_t i = 0; i < PSF::QDim; ++i) c(i) = q_weights(i);
  for (size_t i = 0; i < PSF::YDim; ++i) c(PSF::QDim + i) = y_weights(i);

  return convex::make_weighted_sum(c, Z);
}

/**
 * @brief make quadratic objective over merged PLDI variable
 *
 * obj(z) = 0.5 z^T P z + q^T z
 *
 * @param P quadratic weight over merged z
 * @param q linear weight over merged z
 * @param Z merged decision var
 *
 * @return quadratic objective atom
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_quadratic_objective(
    const linalg::Matrix<ss_type_t<SS>, PolytopicStateFeedback<SS, N>::ZDim,
                         PolytopicStateFeedback<SS, N>::ZDim>& P,
    const linalg::Vector<ss_type_t<SS>, PolytopicStateFeedback<SS, N>::ZDim>& q,
    const ZVar& Z) {
  return convex::make_quad_form(P, q, Z);
}

/**
 * @brief make PLDI problem with objective
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(Q); vec(Y)]
 * @param obj objective atom over merged var
 * @param margin PSD margin on Q block
 *
 * @return PLDI optimization problem
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_problem(
    const linalg::Vector<SS, N>& poly, const ZVar& Z, const auto& obj,
    ss_type_t<SS> margin = 1e-6) {
  const auto q_geq =
      make_polytopic_state_feedback_q_constraint<SS, N>(Z, margin);
  const auto pldi = PolytopicStateFeedback<SS, N>()(poly, Z);
  return convex::make_problem(Z).in_set(q_geq).in_set(pldi).minimize(obj);
}

/**
 * @brief make PLDI feasibility problem
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(Q); vec(Y)]
 * @param margin PSD margin on Q block
 *
 * @return PLDI feasibility problem
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_state_feedback_problem(
    const linalg::Vector<SS, N>& poly, const ZVar& Z,
    ss_type_t<SS> margin = 1e-6) {
  const auto q_geq =
      make_polytopic_state_feedback_q_constraint<SS, N>(Z, margin);
  const auto pldi = PolytopicStateFeedback<SS, N>()(poly, Z);
  return convex::make_problem(Z).in_set(q_geq).in_set(pldi).feasibility();
}

} // namespace control
