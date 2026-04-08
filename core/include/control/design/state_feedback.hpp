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

#include "control/model/state_space.hpp"
#include "convex/sets/lmi.hpp"
#include "convex/variable.hpp"
#include "linalg/operator/matrix_inverse.hpp"

namespace control {
/**
 * @brief StateFeedback
 *
 * Linear Differential Inclusion (LDI) State Feedback
 * Given the LTI system
 *   xdot = A x + B_u u
 *
 * We seek a static state-feedback
 *   u = K x
 *
 * such that the closed-loop system is quadratically stable.
 *
 * There exists P > 0 such that
 *   (A + B_u K)^T P + P (A + B_u K) < 0
 *
 * Equivalently, letting Q = P^{-1} > 0,
 *   Q (A + B_u K)^T + (A + B_u K) Q < 0
 *
 * This is bilinear in Q and K, so define
 *   Y = K Q
 * hence
 *   K = Y Q^{-1}
 *
 * Now \exists \sigma > 0 s.t. Y can be eliminated.
 *   A Q + Q A^T - \sigma B_u B_u^T < 0
 *
 * Assume \sigma = 1 \to A Q + Q A^T - B_u B_u^T < 0
 */
template <ContinuousStateSpaceLike SS>
struct StateFeedback {
  /// @brief extracting alias and types
  using StateSpace = linalg::remove_cvref_t<SS>;

  using T = ss_type_t<StateSpace>;
  using State = state_vec_t<StateSpace>;
  using Input = input_vec_t<StateSpace>;

  static constexpr size_t StateDim = linalg::vec_size_v<State>;
  static constexpr size_t InputDim = linalg::vec_size_v<Input>;

  using QMat = linalg::Matrix<T, StateDim, StateDim>;
  using KMat = linalg::Matrix<T, InputDim, StateDim>;
  using KTMat = linalg::Matrix<T, StateDim, InputDim>;

  static constexpr size_t QDim = StateDim * (StateDim + 1) / 2;
  using FMat = QMat;
  using Fs = linalg::Vector<FMat, QDim>;

  /**
   * @brief operator overload (form state feedback)
   *
   * @param ss (StateSpace)
   * @param Q (Var)
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto operator()(const SS& ss, const Var& Q) const {
    return apply(ss, Q);
  }

  /**
   * @brief apply
   *
   * @param ss (StateSpace)
   * @param Q (Var)
   *
   * @return LDI as LMI
   */
  template <convex::AtomLike Var>
  constexpr auto apply(const SS& ss, const Var& Q) const {
    static_assert(convex::is_var_type_v<Var>, "Input atom is not a var");
    using QDom = convex::meta_dom_t<convex::atom_meta_t<Var>>;
    static_assert(convex::dom_vartype_v<QDom> == convex::VarType::PSD,
                  "Var type != PSD");
    using QSpace = convex::dom_space_t<QDom>;
    static_assert(std::same_as<QSpace, linalg::SymmetricMatrix<QMat>>,
                  "Expected Q var size != Q mat size");

    /// (dakre) note sign change to make this psd
    auto M = [&](const QMat& E) -> FMat {
      return T(-1) * (ss.A * E + E * linalg::transpose(ss.A));
    };

    FMat B = ss.B * linalg::transpose(ss.B);

    linalg::MatrixPencil<FMat, QDim> pencil;
    const auto F_blks = pencil.template apply<QMat>(M);

    return convex::LMI(F_blks, B, Q);
  }

  /**
   * @brief K
   *
   * K = -0.5 B^T Q^{-1}
   * K^T = -0.5 Q^{-T} B
   *
   * @param ss (StateSpace)
   * @param Q (QMat)
   *
   * @return K
   */
  constexpr auto K(const SS& ss, const QMat& Q) const {
    const auto [Q_inv, event] = linalg::SymmetricMatrixInverse<QMat>()(Q);
    if (event != linalg::NumericalEvent::NONE) return std::tuple(KMat{}, event);

    KTMat KT{};
    for (size_t i = 0; i < InputDim; ++i) {
      const auto B_col = linalg::get_col(ss.B, i);
      for (size_t j = 0; j < StateDim; ++j) {
        KT(j)(i) = T(-0.5) *
                   linalg::SymmetricMatrixInverse<QMat>()(linalg::transpose(Q))
                       .apply(B_col);
      }
    }
    return linalg::transpose(KT);
  }
};
}  // namespace control
