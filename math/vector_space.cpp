/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include "algebra/vector_space.hpp"

namespace vector_space {

template <typename T>
topology::MetricSpace<T> to_metric_space(const T& v) {
  topology::MetricSpace<T> metric_space;
  metric_space.tau.open_sets.insert(v.begin(), v.end());
  metric_space.d = [](const T& x, const T& y) { return (x - y).norm(); };
  return metric_space;
}

}  // namespace vector_space
