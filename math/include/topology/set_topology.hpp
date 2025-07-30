/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "topology.hpp"

namespace topology {

template <typename T, size_t N>
struct SetTopology : Topology<SetTopology<T, N>, T> {
  static constexpr size_t s_size = N;

 private:
  friend Topology<SetTopology<T, N>, T>;

  constexpr bool contains_impl(const T& s) const {
    return this->open_sets.find(s) != this->open_sets.end();
  }

  constexpr T interior_impl(const T& X) const {
    T int_X;
    // 1. iterate through all open sets in the topology
    for (const auto& U : this->open_sets) {
      // 2. check if U is fully contained in X (i.e., U \subset X)
      bool is_subset = std::all_of(
          U.begin(), U.end(), [&X](const auto& elem) { return X.count(elem); });
      // 3. add elements of U to the resulting interior set
      if (is_subset) int_X.insert(U.begin(), U.end());
    }
    return int_X;
  }

  constexpr T closure_impl(const T& X) const {
    // Closure of X are all the points of X + all limit points
    // cl(X) = \Intersection \{ C \subset S | C is closed and X \subset C \}
    // The above is the smallest closed set containing X (i.e., X \Union (\all
    // limit points of X))
    // C is closed iff its complement is open (i.e., C^C \in tau)

    // 1. construct the universe S from topology of open sets via union
    T universe;
    for (const auto& U : this->open_sets) universe.insert(U.begin(), U.end());

    // 2. intersect all closed supersets of X by starting with the full universe
    // and iteratively finding the smallest closed set containing X
    auto cl_X = universe;
    for (const auto& U : this->open_sets) {
      T C;  // C = complement of U (C is a closed set)
      for (const auto& x : universe) {
        if (!U.count(x)) C.insert(x);
      }

      // If C is a closed set containing X make the intersection
      auto contains_X = std::all_of(X.begin(), X.end(),
                                    [&C](const auto& x) { return C.count(x); });

      if (contains_X) {
        T new_candidate;
        for (const auto& x : cl_X) {
          if (C.count(x)) new_candidate.insert(x);
        }
        cl_X = std::move(new_candidate);
      }
    }

    return cl_X;
  }

  constexpr T exterior_impl(const T& X) const {
    // 1. construct universe S from topology open sets via union (e.g., all
    // points present in the topological space => S = \Union_{U \in \tau} U)
    // ext(X) = int(X^c)
    T universe;
    for (const auto& U : this->open_sets) universe.insert(U.begin(), U.end());
    // 2. construct the complement of X in the universe S (i.e., X^c = S \ X)
    T complement;
    for (const auto& x : universe) {
      if (!X.count(x)) complement.insert(x);
    }
    // 3. compute the interior of the complement ext(X) = int(X^c)
    return this->interior(complement);
  }

  constexpr T boundary_impl(const T& X) const {
    // Boundary of X lies at the interface between X and X^c
    // bd(X) = cl(X) \ int(X)
    auto cl_X = this->closure(X);
    auto int_X = this->interior(X);

    // 1. construct boundary by selecting elements from the closure of X that
    // are not in the interior of X
    T bd_X;
    for (const auto& x : cl_X) {
      if (!int_X.count(x)) bd_X.insert(x);
    }
    return bd_X;
  }

  constexpr bool is_closed_under_union_impl() const {
    // 1. copy open sets into a vector for indexing
    std::vector<T> set_vector(this->open_sets.begin(), this->open_sets.end());
    size_t n = set_vector.size();

    // 2. loop over all non-empty subsets of open sets using bitmasking
    for (size_t mask = 1; mask < (1 << n); ++mask) {
      T union_set;

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
    for (const auto& A : this->open_sets) {
      for (const auto& B : this->open_sets) {
        T inter;
        std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                              std::inserter(inter, inter.begin()));
        if (!this->contains(inter)) return false;
      }
    }
    return true;
  }

  constexpr bool is_hausdorff_impl() const {
    // 1. construct the universe S from topology of open sets via union
    T universe;
    for (const auto& U : this->open_sets) universe.insert(U.begin(), U.end());

    // 2. test that for all points in the space, two distinct points can be
    // found in two disjoint open sets where one of the points belongs to
    // open set U and the other belongs to open set V
    for (const auto& x : universe) {
      for (const auto& y : universe) {
        if (x == y) continue;

        for (const auto& U : this->open_sets) {
          bool is_disjoint = false;
          for (const auto& V : this->open_sets) {
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
    return this->open_sets.size() < std::numeric_limits<size_t>::max();
  }
};
}  // namespace topology
