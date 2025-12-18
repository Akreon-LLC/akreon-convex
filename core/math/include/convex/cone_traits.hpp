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

/// @brief concept for encapsulating some convex atom
template <class Atom>
concept ConvexAtomLike =
    convex::ConvexFnLike<Atom> || convex::ConcaveFnLike<Atom> ||
    convex::ConvexSetLike<Atom>;

/// @brief cone base class
template <linalg::VectorLike Vec, typename Derived>
struct Cone;

/// @brief forward decl for zero order cone
template <linalg::VectorLike Vec>
struct ZeroOrderCone;

/// @brief forward decl for nonneg orthant cone
template <linalg::VectorLike Vec>
struct NonnegOrthantCone;

/// @brief forward decl for soc
template <linalg::VectorLike Vec>
struct SecondOrderCone;

/// @brief forward decl for psd
template <linalg::VectorLike Vec>
struct PositiveSemidefiniteCone;

/// @brief forward decl for exp
template <linalg::VectorLike Vec>
struct ExpCone;

/// @brief forward decl for power
template <linalg::VectorLike Vec>
struct PowerCone;

/// @brief forward decl for rsoc
template <linalg::VectorLike Vec>
struct RotatedSecondOrderCone;

/// @brief forward decl for log det cone
template <linalg::VectorLike Vec>
struct LogDetCone;

/// @brief cone traits
template <typename Cone>
struct cone_traits;

/// @brief cone traits derived for zoc
template <linalg::VectorLike Vec>
struct cone_traits<ZeroOrderCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for rplus
template <linalg::VectorLike Vec>
struct cone_traits<NonnegOrthantCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for soc
template <linalg::VectorLike Vec>
struct cone_traits<SecondOrderCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for psd
template <linalg::VectorLike Vec>
struct cone_traits<PositiveSemidefiniteCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for exp
template <linalg::VectorLike Vec>
struct cone_traits<ExpCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for power cone
template <linalg::VectorLike Vec>
struct cone_traits<PowerCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for rsoc
template <linalg::VectorLike Vec>
struct cone_traits<RotatedSecondOrderCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief cone traits derived for log det cone
template <linalg::VectorLike Vec>
struct cone_traits<LogDetCone<Vec>> {
  using SlackT = Vec;
  using T = linalg::vec_type_t<Vec>;
  static constexpr size_t N = linalg::vec_size_v<Vec>;
};

/// @brief alias for scalar type in cone
template <class Cone>
using cone_scalar_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::T;

/// @brief alias for slack type in cone
template <class Cone>
using cone_slack_t = typename cone_traits<linalg::remove_cvref_t<Cone>>::SlackT;

/// @brief helper to get N from cone
template <class Cone>
inline constexpr size_t cone_dim_t =
    cone_traits<linalg::remove_cvref_t<Cone>>::N;

/// @brief concept for convex object like
template <class Cone>
concept ConeLike = requires(const Cone& cone, const cone_slack_t<Cone>& s) {
  { cone.barrier(s) };
  { cone.dual_barrier(s) };
  { cone.gradient(s) };
  { cone.dual_gradient(s) };
  { cone.hessian(s) };
  { cone.dual_hessian(s) };
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
   * @param param pack of cones
   */
  constexpr explicit ConeBlock(const Map& map_in, Cs... cs)
      : map(map_in), cones(std::forward<Cs>(cs)...) {}

  /**
   * @brief CTOR for cone block
   *
   * @param affine map
   * @param tuple of cones
   */
  constexpr explicit ConeBlock(const Map& map_in, std::tuple<Cs...> cs)
      : map(map_in), cones(std::move(cs)) {}

  /**
   * @brief validate cone block
   *
   * @return T/F
   */
  constexpr bool validate() const {
    // (dakre) lambda functor to perform the soft check
    size_t n_dims = 0;
    auto perform_check = [&](const auto& arg) {
      using ArgT = cone_scalar_t<decltype(arg)>;
      static_assert(
          std::same_as<ArgT, T>,
          "Cone scalar type is mismatched with cone block scalar type");
      n_dims += cone_dim_t<decltype(arg)>;
    };
    // (dakre) apply fold expression over comma operator on stored cones
    std::apply([&](const auto&... args) { (perform_check(args), ...); }, cones);
    return n_dims == M;
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
