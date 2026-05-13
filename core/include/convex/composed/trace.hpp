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

#include "linalg/operator/operator_traits.hpp"
#include "weighted_sum.hpp"

namespace convex {

/// @brief Forward decl of trace weights
template <typename Space>
struct TraceWeights;

/// @brief TraceWeights for non symmetric square matrix
template <linalg::SquareMatrixLike Mat>
  requires(!linalg::is_symmetric_matrix_space_v<Mat>)
struct TraceWeights<Mat> {
  /// @brief alias and type extractions
  using T = linalg::space_type_t<Mat>;
  static constexpr size_t N = linalg::mat_rows_v<Mat>;
  static constexpr size_t Dim = linalg::space_dim_v<Mat>;
  using Vec = linalg::Vector<T, Dim>;

  /// @brief make helper to form weights for trace objective
  static constexpr auto make() {
    Vec weights{};
    for (size_t i = 0; i < N; ++i) weights(i * N + i) = T(1);
    return weights;
  }
};

/// @brief TraceWeights for symmetric square matrix
template <linalg::SquareMatrixLike Mat>
struct TraceWeights<linalg::SymmetricMatrix<Mat>> {
  /// @brief alias and type extractions
  using Space = linalg::SymmetricMatrix<Mat>;
  using T = linalg::space_type_t<Space>;
  static constexpr size_t N = linalg::mat_rows_v<Mat>;
  static constexpr size_t Dim = linalg::space_dim_v<Space>;
  using Vec = linalg::Vector<T, Dim>;

  /// @brief make helper to form weights for trace objective
  static constexpr auto make() {
    Vec weights{};
    for (size_t i = 0; i < N; ++i) weights(i * (i + 1) / 2 + i) = T(1);
    return weights;
  }
};

/**
 * @brief Trace objective
 *
 * Forms Tr(X) as a linear objective over either a full square matrix variable
 * or a symmetric-matrix variable encoded with svec ordering.
 */
template <linalg::RnSpace Space>
struct Trace {
  /// @brief alias and type extractions
  using T = linalg::space_type_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;
  using Vec = linalg::Vector<T, N>;

  static constexpr auto weights() { return TraceWeights<Space>::make(); }

  /**
   * @brief operator overload
   *
   * @param X (Var)
   *
   * @return WeightedSum Convex Object
   */
  template <AtomLike Var>
  constexpr auto operator()(const Var& X) const {
    static_assert(is_var_type_v<Var>, "Trace must operate on a variable atom");
    static_assert(
        std::same_as<dom_space_t<meta_codom_t<atom_meta_t<Var>>>, Space>,
        "Trace variable codomain must match trace space");

    return WeightedSum<Space>()(weights(), X);
  }
};

/**
 * @brief make trace objective with domain deduced from variable atom.
 *
 * @param X square matrix or symmetric-matrix decision variable
 *
 * @return Tr(X)
 */
template <AtomLike Var>
constexpr auto make_trace(const Var& X) {
  using Space = dom_space_t<meta_codom_t<atom_meta_t<Var>>>;
  return Trace<Space>()(X);
}

/**
 * @brief make trace objective over a matrix block inside a merged variable.
 *
 * @tparam Offset starting scalar coordinate of the traced matrix block
 * @tparam TraceSpace full square matrix space or symmetric-matrix space
 *
 * @param X merged decision variable containing the matrix block
 *
 * @return linear objective equal to the trace of the selected block
 */
template <size_t Offset, linalg::RnSpace TraceSpace, AtomLike Var>
constexpr auto make_trace_block(const Var& X) {
  static_assert(is_var_type_v<Var>, "Trace block must operate on a variable");
  using VarSpace = dom_space_t<meta_codom_t<atom_meta_t<Var>>>;
  using T = linalg::space_type_t<VarSpace>;
  static constexpr size_t VarDim = linalg::space_dim_v<VarSpace>;
  static constexpr size_t TraceDim = linalg::space_dim_v<TraceSpace>;
  static_assert(Offset + TraceDim <= VarDim,
                "Trace block exceeds variable dimension");
  static_assert(std::same_as<T, linalg::space_type_t<TraceSpace>>,
                "Trace block scalar type must match variable scalar type");

  linalg::Vector<T, VarDim> weights{};
  const auto trace_weights = Trace<TraceSpace>::weights();
  for (size_t i = 0; i < TraceDim; ++i) weights(Offset + i) = trace_weights(i);

  return WeightedSum<VarSpace>()(weights, X);
}

} // namespace convex
