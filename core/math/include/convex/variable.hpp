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

namespace convex {

template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct ScalarView : public Atom<Meta, ScalarView<Meta, Child>, Child> {
  using BaseT = Atom<Meta, ScalarView<Meta, Child>, Child>;
  using typename BaseT::Domain;
  using typename BaseT::LinearMatT;
  using typename BaseT::T;
  using typename BaseT::TransT;

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in)
      : BaseT(std::make_tuple(child_in)), idx(idx_in) {}

  constexpr explicit ScalarView(const size_t& idx_in, const Child& child_in,
                                const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(std::make_tuple(child_in), map_in), idx(idx_in) {}

  constexpr void validate_domain_impl(const auto& dom,
                                      const auto& child_codom) const {
    if constexpr (!std::same_as<dom_type_t<decltype(dom)>,
                                dom_type_t<decltype(child_codom)>>)
      throw std::domain_error(
          "Domain type mismatch between variable and scalar view");
  }

  constexpr auto mul_impl(const T& alpha) const {
    const auto mat = linalg::mat_traits<LinearMatT>();
    affine::AffineMap<LinearMatT> new_map(mat(alpha), this->map.translation());
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  constexpr auto add_impl(const T& beta) const {
    affine::AffineMap<LinearMatT> new_map(this->map.linear_part(),
                                          TransT{beta});
    return ScalarView(idx, std::get<0>(this->children), new_map);
  }

  size_t idx;
};

/// @brief variable definition
template <typename Meta>
struct Variable : public Atom<Meta, Variable<Meta>> {
  using BaseT = Atom<Meta, Variable<Meta>>;
  using BaseT::DomSign;
  using BaseT::N;
  using typename BaseT::Domain;
  using typename BaseT::LinearMatT;
  using typename BaseT::T;
  using typename BaseT::TransT;

  constexpr explicit Variable() : BaseT() {}

  constexpr explicit Variable(const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(map_in) {}

  constexpr void validate_domain_impl(const auto& /*dom*/,
                                      const auto& /*child_codom*/) const {
    // no-op
  }

  using ScalarDom = ConvexDomain<linalg::Matrix<T, 1, 1>, DomSign>;
  using ScalarMeta =
      ConvexAtomMeta<ScalarDom, ScalarDom, CurvatureType::AFFINE>;
  using ScalarViewT = ScalarView<ScalarMeta, Variable<Meta>>;
  constexpr std::optional<ScalarViewT> operator()(const size_t& idx) const {
    if (idx >= N) return std::nullopt;
    return ScalarViewT(idx, *this);
  }

  constexpr auto mul_impl(const LinearMatT& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha, this->map.translation());
    return Variable(new_map);
  }

  constexpr auto add_impl(const TransT& beta) const {
    affine::AffineMap<LinearMatT> new_map(this->map.linear_part(),
                                          TransT{beta});
    return Variable(new_map);
  }
};

}  // namespace convex
