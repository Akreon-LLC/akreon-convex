/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <functional>
#include <limits>
#include <set>
#include <vector>

namespace topology {

// Base set X
template <typename T>
using Set = std::set<T>;

// Collection of open sets = topology
template <typename Derived, typename T>
struct Topology {
  using value_type = T;

  constexpr bool contains(const T& set) const {
    static_assert(std::is_invocable_r_v<bool, decltype(&Derived::contains_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->contains_impl(set);
  }

  constexpr T interior(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::interior_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->interior_impl(set);
  }

  constexpr T closure(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::closure_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->closure_impl(set);
  }

  constexpr T exterior(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::exterior_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->exterior_impl(set);
  }

  constexpr T boundary(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::boundary_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->boundary_impl(set);
  }

  constexpr bool is_closed_under_union() const {
    static_assert(
        std::is_invocable_r_v<bool,
                              decltype(&Derived::is_closed_under_union_impl),
                              const Derived*>);
    return static_cast<const Derived*>(this)->is_closed_under_union_impl();
  }

  constexpr bool is_closed_under_finite_intersection() const {
    static_assert(
        std::is_invocable_r_v<
            bool, decltype(&Derived::is_closed_under_finite_intersection_impl),
            const Derived*>);
    return static_cast<const Derived*>(this)
        ->is_closed_under_finite_intersection_impl();
  }

  constexpr bool is_hausdorff() const {
    static_assert(
        std::is_invocable_r_v<bool, decltype(&Derived::is_hausdorff_impl),
                              const Derived*>);
    return static_cast<const Derived*>(this)->is_hausdorff_impl();
  }

  constexpr bool is_second_countable() const {
    static_assert(
        std::is_invocable_r_v<bool,
                              decltype(&Derived::is_second_countable_impl),
                              const Derived*>);
    return static_cast<const Derived*>(this)->is_second_countable_impl();
  }
};

template <typename S>
struct SetTopology : Topology<SetTopology<S>, S> {
  std::set<S> open_sets;

 private:
  friend Topology<SetTopology<S>, S>;

  constexpr bool contains_impl(const S& s) const {
    return open_sets.find(s) != open_sets.end();
  }

  constexpr S interior_impl(const S& X) const {
    S int_X;
    // 1. iterate through all open sets in the topology
    for (const auto& U : open_sets) {
      // 2. check if U is fully contained in X (i.e., U \subset X)
      bool is_subset = std::all_of(
          U.begin(), U.end(), [&X](const auto& elem) { return X.count(elem); });
      // 3. add elements of U to the resulting interior set
      if (is_subset) int_X.insert(U.begin(), U.end());
    }
    return int_X;
  }

  constexpr S closure_impl(const S& X) const {
    // Closure of X are all the points of X + all limit points
    // cl(X) = \Intersection \{ C \subset S | C is closed and X \subset C \}
    // The above is the smallest closed set containing X (i.e., X \Union (\all
    // limit points of X))
    // C is closed iff its complement is open (i.e., C^C \in tau)

    // 1. construct the universe S from topology of open sets via union
    S universe;
    for (const auto& U : open_sets) universe.insert(U.begin(), U.end());

    // 2. intersect all closed supersets of X by starting with the full universe
    // and iteratively finding the smallest closed set containing X
    auto cl_X = universe;
    for (const auto& U : open_sets) {
      S C;  // C = complement of U (C is a closed set)
      for (const auto& x : universe) {
        if (!U.count(x)) C.insert(x);
      }

      // If C is a closed set containing X make the intersection
      auto contains_X = std::all_of(X.begin(), X.end(),
                                    [&C](const auto& x) { return C.count(x); });

      if (contains_X) {
        S new_candidate;
        for (const auto& x : cl_X) {
          if (C.count(x)) new_candidate.insert(x);
        }
        cl_X = std::move(new_candidate);
      }
    }

    return cl_X;
  }

  constexpr S exterior_impl(const S& X) const {
    // 1. construct universe S from topology open sets via union (e.g., all
    // points present in the topological space => S = \Union_{U \in \tau} U)
    // ext(X) = int(X^c)
    S universe;
    for (const auto& U : open_sets) universe.insert(U.begin(), U.end());
    // 2. construct the complement of X in the universe S (i.e., X^c = S \ X)
    S complement;
    for (const auto& x : universe) {
      if (!X.count(x)) complement.insert(x);
    }
    // 3. compute the interior of the complement ext(X) = int(X^c)
    return this->interior(complement);
  }

  constexpr S boundary_impl(const S& X) const {
    // Boundary of X lies at the interface between X and X^c
    // bd(X) = cl(X) \ int(X)
    auto cl_X = this->closure(X);
    auto int_X = this->interior(X);

    // 1. construct boundary by selecting elements from the closure of X that
    // are not in the interior of X
    S bd_X;
    for (const auto& x : cl_X) {
      if (!int_X.count(x)) bd_X.insert(x);
    }
    return bd_X;
  }

  constexpr bool is_closed_under_union_impl() const {
    // 1. copy open sets into a vector for indexing
    std::vector<S> set_vector(open_sets.begin(), open_sets.end());
    size_t n = set_vector.size();

    // 2. loop over all non-empty subsets of open sets using bitmasking
    for (size_t mask = 1; mask < (1 << n); ++mask) {
      S union_set;

      // 3. for each bit in mask, if bit i is set, include set_vector[i]
      for (size_t i = 0; i < n; ++i) {
        if (mask & (1 << i))
          union_set.insert(set_vector[i].begin(), set_vector[i].end());
      }

      // 4. check if this union is itself in the topology tau
      if (!this->contains(union_set)) return false;
    }

    // 5. all unions checked and valid
    return true;
  }

  constexpr bool is_closed_under_finite_intersection_impl() const {
    for (const auto& A : open_sets) {
      for (const auto& B : open_sets) {
        S inter;
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                              std::inserter(inter, inter.begin()));
        if (!this->contains(inter)) return false;
      }
    }
    return true;
  }

  constexpr bool is_hausdorff_impl() const {
    // 1. construct the universe S from topology of open sets via union
    S universe;
    for (const auto& U : open_sets) universe.insert(U.begin(), U.end());

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

  constexpr bool is_second_countable_impl() const {
    // (dakre) TODO have a symbolic approach that tests if the union of finite
    // bases in the topology full span the space, however crude check to ensure
    // topology's open sets are indeed finite is okay for now
    return open_sets.size() < std::numeric_limits<size_t>::max();
  }
};

template <typename Derived, typename T>
concept TopologicalSpace = requires(Topology<Derived, T> tau, T X) {
  { tau.contains(T{}) } -> std::same_as<bool>;  // null set is open
  { tau.contains(X) } -> std::same_as<bool>;    // X is open
  { tau.is_closed_under_union() } -> std::same_as<bool>;
  { tau.is_closed_under_finite_intersection() } -> std::same_as<bool>;
};

}  // namespace topology
