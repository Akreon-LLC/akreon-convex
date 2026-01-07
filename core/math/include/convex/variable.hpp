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
  using typename BaseT::Domain;
  using typename BaseT::T;

  static constexpr size_t N = dom_size_v<meta_dom_t<atom_meta_t<Child>>>;

  using SelMatT = linalg::Matrix<T, 1, N>;
  using TransT = linalg::Vector<T, 1>;

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in)
      : BaseT(std::make_tuple(child_in)),
        map(affine::make_selector_map<SelMatT>(idx_in)),
        idx(idx_in) {
    if (!is_var_type_v<Child>)
      throw std::domain_error(
          "Scalar view child must be the decision variable");
  }

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in,
                                const affine::AffineMap<SelMatT>& map_in)
      : BaseT(std::make_tuple(child_in)), map(map_in), idx(idx_in) {
    if (!is_var_type_v<Child>)
      throw std::domain_error(
          "Scalar view child must be the decision variable");
  }

  constexpr auto mul(const T& alpha) const {
    affine::AffineMap<SelMatT> new_map(alpha * map.linear_part(),
                                       map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  constexpr auto add(const T& beta) const {
    affine::AffineMap<SelMatT> new_map(map.linear_part(),
                                       TransT{beta} + map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto cb = builder.cb;
    return ConeBlockBuilder<builder_dom_t<Builder>, decltype(map),
                            decltype(cb)>(map, cb);
  }

  affine::AffineMap<SelMatT> map;
  size_t idx;
};

/// @brief variable definition
template <typename Meta>
struct Variable : Atom<Meta, Variable<Meta>> {
  using BaseT = Atom<Meta, Variable<Meta>>;
  using BaseT::DomSign;
  using BaseT::N;
  using typename BaseT::Domain;
  using typename BaseT::T;

  using LinearMatT = linalg::Matrix<T, N, N>;
  using TransT = linalg::Vector<T, N>;

  constexpr explicit Variable()
      : BaseT(), map(affine::make_map_id<LinearMatT>()) {}

  constexpr explicit Variable(const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(), map(map_in) {}

  using ScalarDom = ConvexDomain<linalg::Matrix<T, 1, 1>, DomSign>;
  using ScalarMeta =
      ConvexAtomMeta<ScalarDom, ScalarDom, CurvatureType::AFFINE>;
  using ScalarViewT = ScalarView<ScalarMeta, Variable<Meta>>;
  constexpr std::optional<ScalarViewT> operator()(const size_t& idx) const {
    if (idx >= N) return std::nullopt;
    affine::AffineMap<linalg::Matrix<T, 1, N>> scalar_map(
        this->map.linear_part()(idx), this->map.translation()(idx));
    return ScalarViewT(idx, *this, scalar_map);
  }

  constexpr auto mul(const T& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * map.linear_part(),
                                          map.translation());
    return Variable(new_map);
  }

  constexpr auto mul(const LinearMatT& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha * map.linear_part(),
                                          map.translation());
    return Variable(new_map);
  }

  constexpr auto add(const TransT& beta) const {
    affine::AffineMap<LinearMatT> new_map(map.linear_part(),
                                          TransT{beta} + map.translation());
    return Variable(new_map);
  }

  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto cb = builder.cb;
    return ConeBlockBuilder<builder_dom_t<Builder>, decltype(map),
                            decltype(cb)>(map, cb);
  }

  affine::AffineMap<LinearMatT> map;
};

}  // namespace convex
