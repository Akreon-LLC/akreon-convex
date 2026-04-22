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

#include "config/assert.hpp"
#include "control/model/state_space.hpp"
#include "convex/composed/quad_form.hpp"
#include "convex/convex_problem.hpp"
#include "convex/sets/lmi.hpp"
#include "convex/variable.hpp"
#include "linalg/operator/matrix_inverse.hpp"
#include "linalg/operator/matrix_pencil.hpp"

namespace control {

/**
 * @brief Polytopic Luenberger Observer
 *
 * Given
 *   xdot = A(t) x + B(t) u
 *   y    = C(t) x
 *
 * design a state observer
 *   xhatdot = A(t) xhat + B(t) u + L (y - C(t) xhat)
 *
 * s.t. the estimation error
 *   e = x - xhat
 * satisfies
 *   edot = (A(t) - L C(t)) e
 *
 * There exists P > 0 s.t.
 *   A_i^T P + P A_i + W C_i + C_i^T W^T < 0
 *
 * where W = P L, and L = P^{-1} W
 */
template <ContinuousStateSpaceLike SS, size_t N>
struct PolytopicObserver {
  /// @brief extracting alias and types
  using StateSpace = linalg::remove_cvref_t<SS>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using OutputMap = ss_output_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t OutputDim = linalg::mat_rows_v<OutputMap>;

  using PMat = linalg::Matrix<T, StateDim, StateDim>;
  using WMat = linalg::Matrix<T, StateDim, OutputDim>;
  using LMat = linalg::Matrix<T, StateDim, OutputDim>;
  using FMat = PMat;

  static constexpr size_t PDim = StateDim * (StateDim + 1) / 2;
  static constexpr size_t WDim = OutputDim * StateDim;
  static constexpr size_t ZDim = PDim + WDim;

  using ZSpace = linalg::RowVector<T, ZDim>;
  using Fs = linalg::Vector<FMat, ZDim>;
  using Polytope = linalg::Vector<SS, N>;

  /**
   * @brief operator overload (polytopic observer)
   *
   * @param poly (Polytope)
   * @param Z ([svec(P); vec(W)])
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const Polytope& poly, const Var& Z) const {
    return apply(poly, Z);
  }

  /**
   * @brief apply
   *
   * @param poly (Polytope)
   * @param Z ([svec(P); vec(W)])
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const Polytope& poly, const Var& Z) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using ZDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<ZDom> == convex::VarType::FREE,
                  "Merged decision var must be free");
    using ZVarSpace = convex::dom_space_t<ZDom>;
    static_assert(std::same_as<ZVarSpace, ZSpace>,
                  "Expected merged decision size [svec(P); vec(W)]");

    auto make_vertex_lmi = [&]<size_t idx>() {
      const auto& ss = poly(idx);

      auto M_p = [&](const PMat& E) -> FMat {
        return T(-1) * (linalg::transpose(ss.A) * E + E * ss.A);
      };

      auto M_w = [&](const WMat& E) -> FMat {
        return T(-1) *
               (E * ss.C + linalg::transpose(ss.C) * linalg::transpose(E));
      };

      const auto Fp_local =
          linalg::MatrixPencil<FMat, PDim>{}.template apply_sym_mat<PMat>(M_p);
      const auto Fw_local =
          linalg::MatrixPencil<FMat, WDim>{}.template apply_full<WMat>(M_w);

      Fs F_blks{};
      for (size_t i = 0; i < PDim; ++i) F_blks(i) = Fp_local(i);
      for (size_t i = 0; i < WDim; ++i) F_blks(PDim + i) = Fw_local(i);

      return convex::LMI(F_blks, FMat{}, Z);
    };

    auto form_intersection = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return convex::Intersection(make_vertex_lmi.template operator()<Is>()...);
    };

    return form_intersection(std::make_index_sequence<N>{});
  }

  /**
   * @brief L
   *
   * L = P^{-1} W
   *
   * @param P (symmetric Lyapunov variable)
   * @param W (lifted luenberger gain W = P L)
   *
   * @return L
   */
  constexpr auto L(const PMat& P, const WMat& W) const {
    const auto [P_inv, event] = linalg::SymmetricMatrixInverse<PMat>()(P);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(LMat{}, event);
    const auto L = P_inv * W;
    return std::tuple(L, linalg::NumericalEvent::NONE);
  }
};

/**
 * @brief make PLDI P >= margin I LMI over merged z = [svec(P); vec(W)]
 *
 * @param Z (merged decision var)
 * @param margin (PSD margin)
 *
 * @return LMI selecting P-block and constraining it to PSD
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_observer_p_constraint(
    const ZVar& Z, ss_type_t<SS> margin = 1e-6) {
  using T = ss_type_t<SS>;
  using PSF = PolytopicObserver<SS, N>;
  using PMat = typename PSF::PMat;
  using Fs = typename PSF::Fs;

  const auto I = linalg::make_identity<T, PSF::StateDim>();
  const auto Fp =
      linalg::MatrixPencil<PMat, PSF::PDim>{}.template apply_sym_mat<PMat>(
          [](const PMat& E) -> PMat { return E; });

  Fs F_blks{};
  for (size_t i = 0; i < PSF::PDim; ++i) F_blks(i) = Fp(i);
  for (size_t i = 0; i < PSF::WDim; ++i) F_blks(PSF::PDim + i) = PMat{};

  return convex::LMI(F_blks, T(-1) * margin * I, Z);
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
constexpr auto make_polytopic_observer_quadratic_objective(
    const linalg::Matrix<ss_type_t<SS>, PolytopicObserver<SS, N>::ZDim,
                         PolytopicObserver<SS, N>::ZDim>& P,
    const linalg::Vector<ss_type_t<SS>, PolytopicObserver<SS, N>::ZDim>& q,
    const ZVar& Z) {
  using PSF = PolytopicObserver<SS, N>;
  return convex::QuadForm<typename PSF::ZSpace>()(P, q, Z);
}

/**
 * @brief make PLDI problem with objective
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(Q); vec(Y)]
 * @param obj objective atom over merged var
 * @param margin PSD margin on P block
 *
 * @return PLDI optimization problem
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_observer_problem(
    const linalg::Vector<SS, N>& poly, const ZVar& Z, const auto& obj,
    ss_type_t<SS> margin = 1e-6) {
  const auto p_geq = make_polytopic_observer_p_constraint<SS, N>(Z, margin);
  const auto pldi = PolytopicObserver<SS, N>()(poly, Z);
  return convex::make_problem(Z).in_set(p_geq).in_set(pldi).minimize(obj);
}

/**
 * @brief make PLDI feasibility problem
 *
 * @param poly polytope vertices
 * @param Z merged decision var [svec(P); vec(W)]
 * @param margin PSD margin on P block
 *
 * @return PLDI feasibility problem
 */
template <ContinuousStateSpaceLike SS, size_t N, convex::AtomLike ZVar>
constexpr auto make_polytopic_observer_problem(
    const linalg::Vector<SS, N>& poly, const ZVar& Z,
    ss_type_t<SS> margin = 1e-6) {
  const auto p_geq = make_polytopic_observer_p_constraint<SS, N>(Z, margin);
  const auto pldi = PolytopicObserver<SS, N>()(poly, Z);
  return convex::make_problem(Z).in_set(p_geq).in_set(pldi).feasibility();
}

} // namespace control
