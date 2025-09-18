/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <functional>
#include <limits>
#include <set>

#include "../algebra/field.hpp"
#include "open_set.hpp"

namespace topology {

// Collection of open sets = topology
template <typename T, size_t N, typename Derived>
struct Topology {
  using value_type = T;

  static constexpr size_t s_size = N;

  OpenSet<T, N> open_sets;

  template <typename... Sets>
  constexpr explicit Topology(Sets&&... sets)
    requires(sizeof...(Sets) <= N)
      : open_sets{{std::forward<Sets>(sets)...}} {}

  Topology() = delete;

  constexpr bool contains(const T& set) const { return open_sets.count(set); }

  constexpr auto compute_universe() const {
    constexpr size_t K = N * T::s_size;
    using Vt = linalg::scalar_type_t<Topology<T, N, Derived>>;
    OpenSet<Vt, K> universe;

    for (const auto& U : open_sets)
      for (const auto& v : U) universe.insert(v);

    return universe;
  }

  template <typename U>
  constexpr T interior(const U& X) const {
    T int_X;
    // 1. iterate through all open sets in the topology
    for (const auto& sub : open_sets) {
      // 2. check if U is fully contained in X (i.e., U \subset X)
      bool is_subset =
          std::all_of(sub.begin(), sub.end(),
                      [&X](const auto& elem) { return X.count(elem); });
      // 3. add elements of U to the resulting interior set
      if (is_subset) int_X.insert(sub.begin(), sub.end());
    }
    return int_X;
  }

  template <typename U>
  constexpr auto exterior(const U& X) const {
    // 1. construct universe S from topology open sets via union (e.g., all
    // points present in the topological space => S = \Union_{U \in \tau} U)
    // ext(X) = int(X^c)
    const auto universe = compute_universe();

    // 2. construct the complement of X in the universe S (i.e., X^c = S \ X)
    const auto complement = set_complement(universe, X);

    // 3. compute the interior of the complement ext(X) = int(X^c)
    return interior(complement);
  }

  constexpr auto closure(const T& X) const {
    // Closure of X are all the points of X + all limit points
    // cl(X) = \Intersection \{ C \subset S | C is closed and X \subset C \}
    // The above is the smallest closed set containing X (i.e., X \Union (\all
    // limit points of X))
    // C is closed iff its complement is open (i.e., C^C \in tau)

    // 1. construct the universe S from topology of open sets via union
    const auto universe = compute_universe();

    using Vt = linalg::scalar_type_t<Topology<T, N, Derived>>;
    const auto Un = decltype(universe)::s_size;
    const auto Xn = T::s_size;
    OpenSet<Vt, Un> cl_X;
    cl_X = set_union<Vt, Un, Xn, Un>(cl_X, X);

    for (const auto& x : universe) {
      for (const auto& U : open_sets) {
        if (U.count(x)) {
          OpenSet<Vt, 1> singleton{x};
          const auto diff = set_difference(X, singleton);
          const auto inter = set_intersection(U, diff);
          if (inter.size() > 0) cl_X = set_union<Vt, Un, 2, Un>(cl_X, inter);
        }
      }
    }

    return cl_X;
  }

  constexpr auto boundary(const T& X) const {
    // Boundary of X lies at the interface between X and X^c
    // bd(X) = cl(X) \ int(X)
    const auto cl_X = closure(X);
    const auto int_X = interior(X);

    // 1. construct boundary by selecting elements from the closure of X that
    // are not in the interior of X
    using Vt = linalg::scalar_type_t<Topology<T, N, Derived>>;
    const auto Cn = decltype(cl_X)::s_size;
    OpenSet<Vt, Cn> bd_X;
    for (const auto& x : cl_X)
      if (!int_X.count(x)) bd_X.insert(x);

    return bd_X;
  }

  constexpr bool is_closed_under_union() const {
    for (size_t i = 0; i < open_sets.size(); ++i) {
      for (size_t j = i + 1; j < open_sets.size(); ++j) {
        const auto union_set = set_union(open_sets(i), open_sets(j));
        const auto interior_set = interior(union_set);
        if (!std::ranges::equal(interior_set, union_set)) return false;
      }
    }
    return true;
  }

  constexpr bool is_closed_under_finite_intersection() const {
    for (const auto& A : open_sets) {
      for (const auto& B : open_sets) {
        const auto inter = set_intersection(A, B);
        if (!contains(inter)) return false;
      }
    }
    return true;
  }

  constexpr bool is_hausdorff() const {
    // 1. construct the universe S from topology of open sets via union
    const auto universe = compute_universe();

    // 2. test that for all points in the space, two distinct points can be
    // found in two disjoint open sets where one of the points belongs to
    // open set U and the other belongs to open set V
    for (const auto& x : universe) {
      for (const auto& y : universe) {
        if (x == y) continue;

        for (const auto& U : open_sets) {
          bool is_disjoint = false;
          for (const auto& V : open_sets) {
            if (U == V) continue;

            // 3. check if the points are disjoint
            if (U.count(x) && V.count(y)) {
              // 4. now check if the sets intersection is empty
              auto disjoint = std::none_of(
                  U.begin(), U.end(),
                  [&V](const auto& elem) { return V.count(elem) > 0; });
              if (disjoint) {
                is_disjoint = true;
                break;
              }
            }
          }
          if (is_disjoint)
            break;
          else
            return false;
        }
      }
    }
    return true;
  }

  constexpr bool is_second_countable() const {
    // (dakre) TODO have a symbolic approach that tests if the union of finite
    // bases in the topology full span the space, however crude check to ensure
    // topology's open sets are indeed finite is okay for now
    return open_sets.size() < std::numeric_limits<size_t>::max();
  }
};

}  // namespace topology
