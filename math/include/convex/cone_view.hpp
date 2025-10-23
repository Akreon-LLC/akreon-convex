/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "cone_view_base.hpp"

namespace convex {

/**
 * @brief Cone View is an owning cone view of the underlying convex set
 */
template <ConvexConeLike Cone>
struct ConeView : ConeViewBase<ConeView<Cone>> {
  /// @helpers from base class
  using ConeBase = ConeViewBase<ConeView<Cone>>;
  using ConeBase::N;
  using typename ConeBase::Point;
  using typename ConeBase::T;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit ConeView(const Cone& cone_in) : ConeBase(cone_in) {}

  /**
   * @brief contains func (wrapped around cone impl)
   *
   * @param point in space
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const Point& x, T tol = 1e-8) const {
    return this->cone.contains_impl(x, tol);
  }

  /**
   * @brief general inequality for cone K (must be proper)
   *
   * x <=_K y
   *
   * @param Point x \in Space
   * @param Point y \in Space
   *
   * @return T/F
   */
  constexpr std::optional<bool> less_than(const Point& x, const Point& y,
                                          T tol = 1e-8) const {
    if (this->is_proper() == ConeCheck::False) return std::nullopt;
    return contains(y - x, tol);
  }

  /**
   * @brief strict inequality for cone K (must be proper)
   *
   * x <_K y
   *
   * @param Point x \in Space
   * @param Point y \in Space
   *
   * @return T/F
   */
  constexpr std::optional<bool> less_than_strict(const Point& x, const Point& y,
                                                 T tol = 1e-8) const {
    if (this->is_proper() == ConeCheck::False) return std::nullopt;

    const auto delta = y - x;
    const auto eps = 2 * tol;
    if constexpr (std::same_as<Point, linalg::Matrix<T, N, N>>) {
      linalg::Matrix<T, N, N> res;
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i)(j) - x(i)(j) + eps;

      if (!contains(res, tol)) return false;

      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i)(j) - x(i)(j) - eps;

      if (!contains(res, tol)) return false;
    } else {
      const linalg::vec_traits<linalg::Vector<T, N>> vec;
      const auto wiggle = vec.fill(eps);
      if (!contains(delta + wiggle, tol)) return false;
      if (!contains(delta - wiggle, tol)) return false;
    }
    return true;
  }

  /**
   * @brief contains func (wrapped around cone impl)
   *
   * @param (x, t) \in R^n x R
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const std::tuple<Point, T>& x, T tol = 1e-8) const {
    return this->cone.contains_impl(x, tol);
  }

  /**
   * @brief general inequality for cone K (must be proper)
   *
   * x <=_K y
   *
   * @param Point (x, t) \in R^n x R
   * @param Point (y, t) \in R^n x R
   *
   * @return T/F
   */
  constexpr std::optional<bool> less_than(const std::tuple<Point, T>& x_in,
                                          const std::tuple<Point, T>& y_in,
                                          T tol = 1e-8) const {
    if (this->is_proper() == ConeCheck::False) return std::nullopt;
    const auto x = std::get<0>(x_in);
    const auto tx = std::get<1>(x_in);
    const auto y = std::get<0>(y_in);
    const auto ty = std::get<1>(y_in);
    const auto res = y - x;
    const auto t_res = ty - tx;
    return contains(std::make_tuple(res, t_res), tol);
  }

  /**
   * @brief strict inequality for cone K (must be proper)
   *
   * x <_K y
   *
   * @param (x, t) \in R^n x R
   * @param (y, t) \in R^n x R
   *
   * @return T/F
   */
  constexpr std::optional<bool> less_than_strict(
      const std::tuple<Point, T>& x_in, const std::tuple<Point, T>& y_in,
      T tol = 1e-8) const {
    if (this->is_proper() == ConeCheck::False) return std::nullopt;

    const auto x = std::get<0>(x_in);
    const auto tx = std::get<1>(x_in);
    const auto y = std::get<0>(y_in);
    const auto ty = std::get<1>(y_in);
    const auto delta = y - x;
    const auto t_delta = ty - tx;

    const auto eps = 2 * tol;
    if constexpr (std::same_as<Point, linalg::Matrix<T, N, N>>) {
      linalg::Matrix<T, N, N> res;
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i)(j) - x(i)(j) + eps;

      if (!contains(std::make_tuple(res, t_delta), tol)) return false;

      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) res[i][j] = y(i)(j) - x(i)(j) - eps;

      if (!contains(std::make_tuple(res, t_delta), tol)) return false;
    } else {
      const linalg::vec_traits<linalg::Vector<T, N>> vec;
      const auto wiggle = vec.fill(eps);
      if (!contains(std::make_tuple(delta + wiggle, t_delta), tol))
        return false;
      if (!contains(std::make_tuple(delta - wiggle, t_delta), tol))
        return false;
    }
    return true;
  }

  /**
   * @brief get dual view of cone
   *
   * @return dual cone view
   */
  constexpr auto dual_view_impl(const Cone& cone) const {
    return DualConeView<Cone>(cone);
  }
};

}  // namespace convex
