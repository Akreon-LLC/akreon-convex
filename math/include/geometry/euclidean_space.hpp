/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/algebra_traits.hpp"

namespace geometry {

/// @brief alias for Point1d
template <linalg::FieldLike T>
using Point1d = linalg::Vector<T, 1>;

/// @brief alias for Point2d
template <linalg::FieldLike T>
using Point2d = linalg::Vector<T, 2>;

/// @brief alias for Point3d
template <linalg::FieldLike T>
using Point3d = linalg::Vector<T, 3>;

/// @brief alias for R1 space
template <linalg::FieldLike T, size_t Dim>
using R1 = linalg::Vector<Point1d<T>, Dim>;

/// @brief alias for R2 space
template <linalg::FieldLike T, size_t Dim>
using R2 = linalg::Vector<Point2d<T>, Dim>;

/// @brief alias for R3 space
template <linalg::FieldLike T, size_t Dim>
using R3 = linalg::Vector<Point3d<T>, Dim>;

/// @brief alias for Rn space
template <linalg::FieldLike T, size_t N, size_t Dim>
using Rn = linalg::Vector<linalg::Vector<T, N>, Dim>;

}  // namespace geometry
