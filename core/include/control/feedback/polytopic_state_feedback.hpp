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

#include "control/model/state_space.hpp"
#include "convex/intersection.hpp"
#include "convex/sets/lmi.hpp"
#include "convex/variable.hpp"
#include "linalg/operator/matrix_inverse.hpp"
#include "linalg/operator/matrix_pencil.hpp"

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
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using YMat = linalg::Matrix<T, InputDim, StateDim>;
  using FMat = QMat;

  static constexpr size_t QDim = StateDim * (StateDim + 1) / 2;
  static constexpr size_t YDim = InputDim * StateDim;
  static constexpr size_t ZDim = QDim + YDim;

  using ZSpace = linalg::Matrix<T, 1, ZDim>;
  using Fs = linalg::Vector<FMat, ZDim>;
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
      const auto& ss = poly(idx);

      auto M_q = [&](const QMat& E) -> FMat {
        return T(-1) * (ss.A * E + E * linalg::transpose(ss.A));
      };

      auto M_y = [&](const YMat& E) -> FMat {
        return T(-1) *
               (ss.B * E + linalg::transpose(E) * linalg::transpose(ss.B));
      };

      const auto Fq_local =
          linalg::MatrixPencil<FMat, QDim>{}.template apply_sym_mat<QMat>(M_q);
      const auto Fy_local =
          linalg::MatrixPencil<FMat, YDim>{}.template apply_full<YMat>(M_y);

      Fs F_blks{};
      for (size_t i = 0; i < QDim; ++i) F_blks(i) = Fq_local(i);
      for (size_t i = 0; i < YDim; ++i) F_blks(QDim + i) = Fy_local(i);

      return convex::LMI(F_blks, FMat{}, Z);
    };

    auto form_intersection = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return convex::Intersection(make_vertex_lmi.template operator()<Is>()...);
    };

    return form_intersection(std::make_index_sequence<N>{});
  }
};

}  // namespace control
