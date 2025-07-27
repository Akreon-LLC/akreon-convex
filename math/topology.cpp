/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "topology/topology.hpp"

#include <algorithm>
#include <limits>

namespace topology {

template <typename T>
bool Topology<T>::contains(const Set<T>& s) const {
  return open_sets.find(s) != open_sets.end();
}

template <typename T>
Set<T> Topology<T>::interior(const Set<T>& X) {
  Set<T> int_X;
  // 1. iterate through all open sets in the topology
  for (const auto& U : open_sets) {
    // 2. check if U is fully contained in X (i.e., U \subset X)
    bool is_subset = std::all_of(U.begin(), U.end(),
                                 [&X](const T& elem) { return X.count(elem); });
    // 3. add elements of U to the resulting interior set
    if (is_subset) int_X.insert(U.begin(), U.end());
  }
  return int_X;
}

template <typename T>
Set<T> Topology<T>::closure(const Set<T>& X) {
  // Closure of X are all the points of X + all limit points
  // cl(X) = \Intersection \{ C \subset S | C is closed and X \subset C \}
  // The above is the smallest closed set containing X (i.e., X \Union (\all
  // limit points of X))
  // C is closed iff its complement is open (i.e., C^C \in tau)

  // 1. construct the universe S from topology of open sets via union
  Set<T> universe;
  for (const auto& U : open_sets) universe.insert(U.begin(), U.end());

  // 2. intersect all closed supersets of X by starting with the full universe
  // and iteratively finding the smallest closed set containing X
  auto cl_X = universe;
  for (const auto& U : open_sets) {
    Set<T> C;  // C = complement of U (C is a closed set)
    for (const auto& x : universe) {
      if (!U.count(x)) C.insert(x);
    }

    // If C is a closed set containing X make the intersection
    auto contains_X = std::all_of(X.begin(), X.end(),
                                  [&C](const T& x) { return C.count(x); });

    if (contains_X) {
      Set<T> new_candidate;
      for (const auto& x : cl_X) {
        if (C.count(x)) new_candidate.insert(x);
      }
      cl_X = std::move(new_candidate);
    }
  }

  return cl_X;
}

template <typename T>
Set<T> Topology<T>::exterior(const Set<T>& X) {
  // 1. construct universe S from topology open sets via union (e.g., all
  // points present in the topological space => S = \Union_{U \in \tau} U)
  // ext(X) = int(X^c)
  Set<T> universe;
  for (const auto& U : open_sets) universe.insert(U.begin(), U.end());
  // 2. construct the complement of X in the universe S (i.e., X^c = S \ X)
  Set<T> complement;
  for (const auto& x : universe) {
    if (!X.count(x)) complement.insert(x);
  }
  // 3. compute the interior of the complement ext(X) = int(X^c)
  return interior(complement);
}

template <typename T>
Set<T> Topology<T>::boundary(const Set<T>& X) {
  // Boundary of X lies at the interface between X and X^c
  // bd(X) = cl(X) \ int(X)
  auto cl_X = closure(X);
  auto int_X = interior(X);

  // 1. construct boundary by selecting elements from the closure of X that
  // are not in the interior of X
  Set<T> bd_X;
  for (const auto& x : cl_X) {
    if (!int_X.count(x)) bd_X.insert(x);
  }
  return bd_X;
}

template <typename T>
  requires TopologicalSpace<T>
bool MetricSpace<T>::distance_metric_is_valid() const {
  Set<T> universe;
  for (const auto& U : tau.open_sets) universe.insert(U.begin(), U.end());

  for (const auto& x : universe) {
    for (const auto& y : universe) {
      T dxy = d(x, y);
      T dyx = d(y, x);

      if (dxy < 0) return false;
      if (x == y && dxy != 0) return false;
      if (x != y && dxy == 0) return false;
      if (std::abs(dxy - dyx) > 1e-9) return false;
      for (const auto& z : universe)
        if (d(x, z) > d(x, y) + d(y, z) + 1e-9) return false;
    }
  }
  return true;
}

template <typename T>
bool is_closed_under_union(const Topology<T>& tau) {
  // 1. copy open sets into a vector for indexing
  const auto& sets = tau.open_sets;
  std::vector<Set<T>> set_vector(sets.begin(), sets.end());
  size_t n = set_vector.size();

  // 2. loop over all non-empty subsets of open sets using bitmasking
  for (size_t mask = 1; mask < (1 << n); ++mask) {
    Set<T> union_set;

    // 3. for each bit in mask, if bit i is set, include set_vector[i]
    for (size_t i = 0; i < n; ++i) {
      if (mask & (1 << i))
        union_set.insert(set_vector[i].begin(), set_vector[i].end());
    }

    // 4. check if this union is itself in the topology tau
    if (!tau.contains(union_set)) return false;
  }

  // 5. all unions checked and valid
  return true;
}

template <typename T>
bool is_closed_under_finite_intersection(const Topology<T>& tau) {
  for (const auto& A : tau.open_sets) {
    for (const auto& B : tau.open_sets) {
      Set<T> inter;
      std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                            std::inserter(inter, inter.begin()));
      if (!tau.contains(inter)) return false;
    }
  }
  return true;
}

template <typename T>
bool is_hausdorff(const Topology<T>& tau) {
  // 1. construct the universe S from topology of open sets via union
  Set<T> universe;
  for (const auto& U : tau.open_sets) universe.insert(U.begin(), U.end());

  // 2. test that for all points in the space, two distinct points can be
  // found in two disjoint open sets where one of the points belongs to
  // open set U and the other belongs to open set V
  for (const auto& x : universe) {
    for (const auto& y : universe) {
      if (x == y) continue;

      for (const auto& U : tau.open_sets) {
        bool is_disjoint = false;
        for (const auto& V : tau.open_sets) {
          if (U == V) continue;

          // 3. check if the points are disjoint
          if (U.count(x) && V.count(y)) {
            // 4. now check if the sets intersection is empty
            auto disjoint =
                std::none_of(U.begin(), U.end(),
                             [&V](const T& elem) { return V.count(elem) > 0; });
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

template <typename T>
bool is_second_countable(const Topology<T>& tau) {
  // (dakre) TODO have a symbolic approach that tests if the union of finite
  // bases in the topology full span the space, however crude check to ensure
  // topology's open sets are indeed finite is okay for now
  return tau.open_sets.size() < std::numeric_limits<size_t>::max();
}

template <typename T>
bool is_locally_euclidean(const Topology<T>& tau) {
  auto d = [](const T& x, const T& y) { return std::abs(x - y); };
  MetricSpace<T> ms{.tau = tau, .d = d};
  return ms.distance_metric_is_valid();
}

template struct Topology<double>;
template struct Topology<float>;
template struct Topology<int>;
template struct Topology<unsigned>;

}  // namespace topology
