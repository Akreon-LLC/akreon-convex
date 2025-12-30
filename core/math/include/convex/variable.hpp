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

#include <optional>
#include <tuple>

#include "convex_traits.hpp"
#include "convex_util.hpp"

namespace convex {

/// @brief alias for scalar var domain
template <linalg::RnSpace Dom>
using FreeVarDom = ConvexDomain<Dom, SignType::FREE>;

/// @brief alias for scalar var meta
template <linalg::RnSpace Dom>
using FreeVarMeta =
    ConvexAtomMeta<FreeVarDom<Dom>, FreeVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for scalar var domain
template <linalg::RnSpace Dom>
using NonnegVarDom = ConvexDomain<Dom, SignType::NONNEG>;

/// @brief alias for scalar var meta
template <linalg::RnSpace Dom>
using NonnegVarMeta =
    ConvexAtomMeta<NonnegVarDom<Dom>, NonnegVarDom<Dom>, CurvatureType::AFFINE>;

/// @brief alias for scalar var domain
template <linalg::RnSpace Dom>
using PosVarDom = ConvexDomain<Dom, SignType::POS>;

/// @brief alias for scalar var meta
template <linalg::RnSpace Dom>
using PosVarMeta =
    ConvexAtomMeta<PosVarDom<Dom>, PosVarDom<Dom>, CurvatureType::AFFINE>;

template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct ScalarView : Atom<Meta, ScalarView<Meta, Child>, Child> {
  using BaseT = Atom<Meta, ScalarView<Meta, Child>, Child>;
  using BaseT::N;
  using typename BaseT::Domain;
  using typename BaseT::LinearMatT;
  using typename BaseT::T;
  using typename BaseT::TransT;

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in)
      : BaseT(
            std::make_tuple(child_in),
            affine::AffineMap(as_row_vector(child_in.map.linear_part()(idx_in)),
                              child_in.map.translation())),
        idx(idx_in) {
    if (!is_var_type_v<Child>)
      throw std::domain_error(
          "Scalar view child must be the decision variable");
  }

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in,
                                const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(std::make_tuple(child_in), map_in), idx(idx_in) {
    if (!is_var_type_v<Child>)
      throw std::domain_error(
          "Scalar view child must be the decision variable");
  }

  constexpr auto mul_impl(const T& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * this->map.linear_part(),
                                          this->map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  constexpr auto add_impl(const T& beta) const {
    affine::AffineMap<LinearMatT> new_map(
        this->map.linear_part(), TransT{beta} + this->map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  template <linalg::RnSpace DecVarDom>
  constexpr auto canonicalize_impl(
      const ConeMapBuilder<DecVarDom>& /*builder*/) const {
    const auto cb = create_empty_cblk<T>();
    return std::tuple(this->map, cb);
  }

  size_t idx;
};

/// @brief variable definition
template <typename Meta>
struct Variable : Atom<Meta, Variable<Meta>> {
  using BaseT = Atom<Meta, Variable<Meta>>;
  using BaseT::DomSign;
  using BaseT::N;
  using typename BaseT::Domain;
  using typename BaseT::LinearMatT;
  using typename BaseT::T;
  using typename BaseT::TransT;

  constexpr explicit Variable() : BaseT(affine::make_map_id<LinearMatT>()) {}

  constexpr explicit Variable(const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(map_in) {}

  using ScalarDom = ConvexDomain<linalg::Matrix<T, 1, 1>, DomSign>;
  using ScalarMeta =
      ConvexAtomMeta<ScalarDom, ScalarDom, CurvatureType::AFFINE>;
  using ScalarViewT = ScalarView<ScalarMeta, Variable<Meta>>;
  constexpr std::optional<ScalarViewT> operator()(const size_t& idx) const {
    if (idx >= N) return std::nullopt;
    return ScalarViewT(idx, *this);
  }

  constexpr auto mul_impl(const T& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * this->map.linear_part(),
                                          this->map.translation());
    return Variable(new_map);
  }

  constexpr auto mul_impl(const LinearMatT& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * this->map.linear_part(),
                                          this->map.translation());
    return Variable(new_map);
  }

  constexpr auto add_impl(const TransT& beta) const {
    affine::AffineMap<LinearMatT> new_map(
        this->map.linear_part(), TransT{beta} + this->map.translation());
    return Variable(new_map);
  }

  template <linalg::RnSpace DecVarDom>
  constexpr auto canonicalize_impl(
      const ConeMapBuilder<DecVarDom>& /*builder*/) const {
    const auto cb = create_empty_cblk<T>();
    return std::tuple(this->map, cb);
  }
};

}  // namespace convex
