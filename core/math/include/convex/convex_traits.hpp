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

namespace convex {

enum struct DomInfo : uint8_t {
  FREE = 0,
  NONNEG,
  NONPOS,
  POS,
  NEG,
  PSD,
  NSD,
  PD,
  ND
};

enum struct Curvature : uint8_t { AFFINE = 0, CONVEX, CONCAVE };

enum struct RangeInfo : uint8_t { UNKNOWN = 0, NONNEG, NONPOS, POS, NEG };

enum struct Monotonicity : uint8_t { UNKNOWN = 0, NONDEC, NONINC };

enum struct ConstraintType : uint8_t { EQ = 0, LEQ, GEQ, IN_SET };

template <linalg::RnSpace Space>
struct ConvexDomain {
  constexpr explicit ConvexDomain(const DomInfo& info_in) : info(info_in) {}
  DomInfo info;

  constexpr bool operator==(const ConvexDomain<Space>& other) const {
    return info == other.info;
  }
};

template <class Dom>
struct dom_traits {};

template <linalg::RnSpace Space>
struct dom_traits<ConvexDomain<Space>> {
  using T = linalg::space_type_t<Space>;
  using Point = linalg::space_point_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;
};

template <class Dom>
using dom_type_t = typename dom_traits<linalg::remove_cvref_t<Dom>>::T;

template <class Dom>
using dom_point_t = typename dom_traits<linalg::remove_cvref_t<Dom>>::Point;

template <class Dom>
inline constexpr size_t dom_size_v = dom_traits<linalg::remove_cvref_t<Dom>>::N;

template <class Dom>
concept ConvexDomainLike = requires(const Dom& dom) {
  { dom.info };
};

template <class Atom>
concept ConvexAtomLike = requires(const Atom& atom) {
  { atom.domain };
  { atom.curvature };
  { atom.range };
  { atom.map };
};

template <ConvexDomainLike Domain, typename Derived, ConvexAtomLike... Cs>
struct Atom;

template <ConvexDomainLike Domain>
struct Variable;

template <ConvexDomainLike Domain>
  requires(dom_size_v<Domain> == 1)
struct ScalarView;

}  // namespace convex
