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

#include "convex_traits.hpp"

namespace convex {

template <ConvexDomainLike Domain, typename Derived, ConvexAtomLike... Cs>
struct Atom {
  using T = dom_type_t<Domain>;
  using Point = dom_point_t<Domain>;
  static constexpr size_t N = dom_size_v<Domain>;
  using LinearMatT = linalg::Matrix<T, N, N>;
  using TransT = linalg::Vector<T, N>;

  constexpr explicit Atom(const Domain& dom_in, const Curvature& curv_in,
                          const RangeInfo& range_in)
      : domain(dom_in), curvature(curv_in), range(range_in) {}

  constexpr explicit Atom(const Domain& dom_in, const Curvature& curv_in,
                          const RangeInfo& range_in,
                          const std::tuple<Cs...> child_in)
      : domain(dom_in),
        curvature(curv_in),
        range(range_in),
        children(std::move(child_in)) {}

  constexpr explicit Atom(const Domain& dom_in, const Curvature& curv_in,
                          const RangeInfo& range_in,
                          const affine::AffineMap<LinearMatT>& map_in)
      : domain(dom_in), curvature(curv_in), range(range_in), map(map_in) {}

  constexpr explicit Atom(const Domain& dom_in, const Curvature& curv_in,
                          const RangeInfo& range_in,
                          const std::tuple<Cs...> child_in,
                          const affine::AffineMap<LinearMatT>& map_in)
      : domain(dom_in),
        curvature(curv_in),
        range(range_in),
        children(std::move(child_in)),
        map(map_in) {}

  constexpr auto mul(const LinearMatT& alpha) const {
    return self().mul_impl(alpha);
  }

  constexpr auto add(const TransT& beta) const { return self().add_impl(beta); }

  constexpr auto mul(const T& alpha) const { return self().mul_impl(alpha); }

  constexpr auto add(const T& beta) const { return self().add_impl(beta); }

  constexpr auto canonicalize() const { return self().canonicalize_impl(); }

  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  Domain domain;
  Curvature curvature;
  RangeInfo range;
  std::tuple<Cs...> children;
  affine::AffineMap<LinearMatT> map;
};

}  // namespace convex
