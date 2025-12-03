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

#include "affine/affine_traits.hpp"
#include "convex_fn_traits.hpp"
#include "convex_set_traits.hpp"

namespace conic {

/// @brief cone base class
template <linalg::MatrixLike Mat, typename Derived>
struct Cone {
  /// @brief alias for mat type
  using T = linalg::mat_type_t<Mat>;

  /// @brief helpers to get M and N sizes
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;

  /// @brief alias for bias vector
  using Bias = linalg::Vector<T, M>;

  /**
   * @brief compute barrier
   *
   * @return barrier for cone
   */
  constexpr auto barrier(const Mat& F, const Bias& g) const {
    return self().barrier_impl(F, g);
  }

  /**
   * @brief compute gradient
   *
   * @return gradient for cone
   */
  constexpr auto gradient(const Mat& F, const Bias& g) const {
    return self().gradient_impl(F, g);
  }

  /**
   * @brief compute hessian
   *
   * @return hessian for cone
   */
  constexpr auto hessian(const Mat& F, const Bias& g) const {
    return self().hessian_impl(F, g);
  }

  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }
};

/// @brief forward decl for zero order cone
template <linalg::MatrixLike Mat>
struct ZeroOrderCone;

/// @brief forward decl for nonneg orthant cone
template <linalg::MatrixLike Mat>
struct NonnegOrthantCone;

/// @brief forward decl for soc
template <linalg::MatrixLike Mat>
struct SecondOrderCone;

/// @brief forward decl for psd
template <linalg::MatrixLike Mat>
struct PositiveSemidefiniteCone;

/// @brief cone traits
template <typename Cone>
struct cone_traits;

/// @brief cone traits derived for zoc
template <linalg::MatrixLike Mat>
struct cone_traits<ZeroOrderCone<Mat>> {
  using T = linalg::mat_type_t<Mat>;
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;
  using Domain = linalg::Vector<T, N>;
  using Bias = linalg::Vector<T, M>;
  using LinearMap = Mat;
};

/// @brief cone traits derived for rplus
template <linalg::MatrixLike Mat>
struct cone_traits<NonnegOrthantCone<Mat>> {
  using T = linalg::mat_type_t<Mat>;
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;
  using Domain = linalg::Vector<T, N>;
  using Bias = linalg::Vector<T, M>;
  using LinearMap = Mat;
};

/// @brief cone traits derived for soc
template <linalg::MatrixLike Mat>
struct cone_traits<SecondOrderCone<Mat>> {
  using T = linalg::mat_type_t<Mat>;
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;
  using Domain = linalg::Vector<T, N>;
  using Bias = linalg::Vector<T, M>;
  using LinearMap = Mat;
};

/// @brief cone traits derived for psd
template <linalg::MatrixLike Mat>
struct cone_traits<PositiveSemidefiniteCone<Mat>> {
  using T = linalg::mat_type_t<Mat>;
  static constexpr size_t M = linalg::mat_rows_v<Mat>;
  static constexpr size_t N = linalg::mat_cols_v<Mat>;
  using Domain = linalg::Vector<T, N>;
  using Bias = linalg::Vector<T, M>;
  using LinearMap = Mat;
};

/// @brief alias for scalar type in cone
template <class Cone>
using cone_type_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::T;

/// @brief alias for domain in cone
template <class Cone>
using cone_domain_t =
    typename cone_traits<linalg::remove_cvref_t<Cone>>::Domain;

/// @brief alias for bias in cone
template <class Cone>
using cone_bias_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::Bias;

/// @brief alias for linear mat in cone
template <class Cone>
using cone_mat_t =
    typename cone_traits<linalg::remove_cvref_t<Cone>>::LinearMap;

/// @brief concept for convex object like
template <class Cone>
concept ConeLike = requires(const Cone& cone, const cone_mat_t<Cone>& F,
                            const cone_bias_t<Cone>& g) {
  { cone.barrier(F, g) };
  { cone.gradient(F, g) };
  { cone.hessian(F, g) };
};

/**
 * @brief Cone block
 *
 * Houses the convex object -> cone block structure
 */
template <affine::AffineMapLike Map, ConeLike... Cs>
struct ConeBlock {
  /// @brief alias to extract scalar type from space
  using MatT = affine::affine_mat_t<Map>;
  using T = linalg::mat_type_t<MatT>;

  /// @brief alias to extract codom & dom of affine map
  static constexpr size_t M = linalg::mat_rows_v<MatT>;
  static constexpr size_t N = linalg::mat_cols_v<MatT>;

  /**
   * @brief CTOR for cone block
   *
   * @param affine map (takes convex object -> cone coordinates (homogenized)
   * @param tag for is primal vs. dual
   */
  constexpr explicit ConeBlock(const Map& map_in, Cs... cs)
      : map(map_in), cones(std::forward<Cs>(cs)...) {}

  /**
   * @brief validate cone block
   *
   * @return T/F
   */
  constexpr bool validate() const {
    // (dakre) lambda functor to perform the soft check
    size_t n_rows = 0;
    auto perform_check = [&](const auto& arg) {
      using ArgT = cone_type_t<decltype(arg)>;
      using ArgMat = cone_mat_t<decltype(arg)>;
      static_assert(std::same_as<ArgT, T>,
                    "Cone type is mismatched with cone block scalar type");
      static_assert(
          N == linalg::mat_cols_v<ArgMat>,
          "Cone domain dim is mismatched with domain vector in cone block");
      n_rows += linalg::mat_rows_v<ArgMat>;
    };
    // (dakre) apply fold expression over comma operator on stored cones
    std::apply([&](const auto&... args) { (perform_check(args), ...); }, cones);
    return n_rows == M;
  }

  /** @brief member vars for cone block */
  Map map;                  ///< affine map taking [x;t] -> cone coords
  std::tuple<Cs...> cones;  ///< tuple of cones aligned with map
};

/// @brief concept for cone block like used in canonical form
template <class Cblk>
concept ConeBlockLike = requires(const Cblk& cblk) {
  { cblk.map };
  { cblk.cones };
};

/**
 * @brief Canonical Form
 *
 * Houses a tuple of cones along with the linear weighting vector for the
 * objective
 *
 * min. c^T x
 * s.t. F z + g \in K = {K_0 x K_1 x ... K_{m}}
 */
template <linalg::RnSpace Domain, ConeBlockLike... Cblks>
struct CanonicalForm {
  /// @brief alias to extract type traits from domain
  using T = linalg::space_type_t<Domain>;
  using Point = linalg::space_point_t<Domain>;

  /// @brief helper to get dim of point & cone blocks
  static constexpr size_t N = linalg::vec_size_v<Point>;
  static constexpr size_t M = sizeof...(Cblks);

  /**
   * @brief CTOR for CanonicalForm
   *
   * @param param pack of cone blocks
   */
  constexpr explicit CanonicalForm(const Point& c_in, Cblks... cblks)
      : cone_blocks(std::forward<Cblks>(cblks)...) {
    for (size_t i = 0; i < N; ++i) c[i] = c_in(i);
    c[N] = T(1);
  }

  /** @brief member vars for canonical form */
  linalg::Vector<T, N + 1> c;  //< linear weight on objective c^T [x;t] where
                               // t is the last lift var
  std::tuple<Cblks...> cone_blocks;  //< K = k_0 x k_1 x ... k_{m} cones
};
}  // namespace conic
