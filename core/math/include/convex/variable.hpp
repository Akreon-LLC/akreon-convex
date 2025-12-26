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

template <ConvexDomainLike Domain>
  requires(dom_size_v<Domain> == 1)
struct ScalarView : public Atom<Domain, ScalarView<Domain>> {
  using BaseT = Atom<Domain, ScalarView<Domain>>;
  using typename BaseT::LinearMatT;
  using typename BaseT::T;
  using typename BaseT::TransT;

  constexpr explicit ScalarView(const size_t& idx_in, const Domain& dom_in,
                                const Curvature& curv_in,
                                const RangeInfo& range_in,
                                const Variable<Domain>& child_in)
      : BaseT(dom_in, curv_in, range_in, children(child_in)), idx(idx_in) {}

  constexpr explicit ScalarView(const size_t& idx_in, const Domain& dom_in,
                                const Curvature& curv_in,
                                const RangeInfo& range_in,
                                const Variable<Domain>& child_in,
                                const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(dom_in, curv_in, range_in, children(std::make_tuple(child_in)),
              map_in),
        idx(idx_in) {}

  constexpr auto mul_impl(const T& alpha) const {
    const auto mat = linalg::mat_traits<LinearMatT>();
    affine::AffineMap<LinearMatT> new_map(mat(alpha), this->map.translation());
    return ScalarView(idx, this->domain, this->curvature, this->range,
                      std::get<0>(this->children), new_map);
  }

  constexpr auto add_impl(const T& beta) const {
    affine::AffineMap<LinearMatT> new_map(this->map.linear_part(),
                                          TransT{beta});
    return ScalarView(idx, this->domain, this->curvature, this->range,
                      std::get<0>(this->children), new_map);
  }

  size_t idx;
};

/// @brief variable definition
template <ConvexDomainLike Domain>
struct Variable : public Atom<Domain, Variable<Domain>> {
  using BaseT = Atom<Domain, Variable<Domain>>;
  using BaseT::N;
  using typename BaseT::LinearMatT;
  using typename BaseT::T;
  using typename BaseT::TransT;

  constexpr explicit Variable(const Domain& dom_in)
      : BaseT(dom_in, Curvature::AFFINE, get_range(dom_in)) {}

  constexpr explicit Variable(const Domain& dom_in,
                              const affine::AffineMap<LinearMatT>& map_in)
      : BaseT(dom_in, Curvature::AFFINE, get_range(dom_in), map_in) {}

  constexpr RangeInfo get_range(const Domain& dom) {
    switch (dom.info) {
      case DomInfo::NONNEG:
        return RangeInfo::NONNEG;
      case DomInfo::NONPOS:
        return RangeInfo::NONPOS;
      case DomInfo::POS:
        return RangeInfo::POS;
      case DomInfo::NEG:
        return RangeInfo::NEG;
      case DomInfo::FREE:
      case DomInfo::PSD:
      case DomInfo::NSD:
      case DomInfo::PD:
      case DomInfo::ND:
      default:
        return RangeInfo::UNKNOWN;
    }
  }

  using ScalarDom = ScalarView<ConvexDomain<linalg::Matrix<T, 1, 1>>>;
  constexpr std::optional<ScalarDom> operator()(const size_t& idx) const {
    if (idx >= N) return std::nullopt;
    const auto dom = ConvexDomain<linalg::Matrixd<1, 1>>(this->domain.info);
    return ScalarView(idx, dom, this->curvature, this->range, *this);
  }

  constexpr auto mul_impl(const LinearMatT& alpha) const {
    affine::AffineMap<LinearMatT> new_map(alpha, this->map.translation());
    return Variable(this->domain, new_map);
  }

  constexpr auto add_impl(const TransT& beta) const {
    affine::AffineMap<LinearMatT> new_map(this->map.linear_part(),
                                          TransT{beta});
    return Variable(this->domain, new_map);
  }
};

}  // namespace convex
