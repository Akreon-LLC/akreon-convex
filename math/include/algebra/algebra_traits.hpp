/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <type_traits>

namespace linalg {

/// @brief forward declaration for vector
template <typename T, size_t N>
struct Vector;

/// @brief alias for matrix definition
template <typename T, size_t Rows, size_t Cols>
using Matrix = Vector<Vector<T, Cols>, Rows>;

/// @brief concrete matrix type traits to separate vector from matrix at compile
/// time
template <typename T>
struct is_matrix : std::false_type {};

template <typename T, size_t Rows, size_t Cols>
struct is_matrix<Matrix<T, Rows, Cols>> : std::true_type {};

/// @brief base type traits definition for vector
template <class Vec>
struct vec_traits;

/// @brief type traits definition for vector
template <typename T, size_t N>
struct vec_traits<Vector<T, N>> {
  using value_type = T;
  static constexpr size_t SIZE = N;
};

/// @brief helper for size of vector
template <class Vec>
inline constexpr size_t vec_size_v = vec_traits<Vec>::SIZE;

/// @brief alias for underlying type for vector
template <class Vec>
using vec_type_t = typename vec_traits<Vec>::value_type;

/// @brief base type traits definition for matrix
template <class Mat>
struct mat_traits;

/// @brief type traits definition for matrix
template <typename T, size_t Rows, size_t Cols>
struct mat_traits<Matrix<T, Rows, Cols>> {
  using value_type = T;
  static constexpr size_t ROWS = Rows;
  static constexpr size_t COLS = Cols;

  template <typename... Ts>
  static constexpr auto make(Ts&&... xs) {
    static_assert(sizeof...(xs) == Rows * Cols);
    Matrix<T, Rows, Cols> M;
    int i = 0;
    auto set_elem = [&](const T& x) {
      const auto col = i % Cols;
      const auto row = i / Cols;
      M[row][col] = x;
    };
    ((void)set_elem(std::forward<Ts>(xs)), ...);
    return M;
  }
};

/// @brief alias for matrix type
template <class Mat>
using mat_type_t = typename mat_traits<Mat>::value_type;

/// @brief helper for size of matrix rows
template <class Mat>
inline constexpr size_t mat_rows_v = mat_traits<Mat>::ROWS;

/// @brief helper for size of matrix cols
template <class Mat>
inline constexpr size_t mat_cols_v = mat_traits<Mat>::COLS;

/** \struct scalar_type
 *  \brief Structure that deals in type_trait deduction for a scalar type
 *  \details scalar_type is an overloaded type that strips away
 * meta-templates to the final primitive scalar type
 *  \ingroup linalg
 */
template <typename T>
struct scalar_type {
  using value_type = T;
};

/// @brief True if type is of a std::array<T, N>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t N>
struct scalar_type<std::array<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a std::complex<T>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T>
struct scalar_type<std::complex<T>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a Vector<T, N>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t N>
struct scalar_type<Vector<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief True if type is of a Vector<Vector<T, N>, M>
/// @tparam value_type is \equiv to the deduced scalar type value
template <typename T, size_t M, size_t N>
struct scalar_type<Vector<Vector<T, M>, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

/// @brief scalar_type_t alias for deducing scalar type of templated argument
template <typename T>
using scalar_type_t = typename scalar_type<T>::value_type;

/** \struct is_complex_t
 *  \brief Structure that deals in type_trait deduction for checking if type
 * is complex (false initially)
 *  \details scalar_type is an overloaded type that strips away meta-templates
 * to see if the type is complex
 *  \ingroup linalg
 */
template <typename T>
struct is_complex_t : public std::false_type {};

/// @brief True if type is of a complex<T>
template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

/// @brief True if type is of a complex<T> otherwise false (default)
template <typename T, size_t N>
struct is_complex_t<Vector<T, N>> {
  static const bool value = is_complex_t<T>::value;
};

/// @brief True if type is of a complex<T> otherwise false (default)
template <typename T, size_t N, size_t M>
struct is_complex_t<Vector<Vector<T, M>, N>> {
  static const bool value = is_complex_t<T>::value;
};

/// @brief is_complex_v is the variable holding the value type of is_complex_t
template <typename T>
inline constexpr bool is_complex_v = is_complex_t<T>::value;

/**
 * @brief FieldLike concept is obtained by floating point values
 *
 * ** Group axioms **
 * 0. a + b \exists
 * 1. (a + b) + c == a + (b + c)
 * 2. -1 * a \exists
 * 3. a + b == b + a
 *
 * ** Ring axioms **
 * Group plus the following
 * 0. a * b \exists
 * 1. (a * b) * c == a * (b * c)
 * 2. a * (b + c) == a * b + a * c
 * 3. a * b == b * a
 *
 * ** Field axioms **
 * 0. a / a == 1
 * 1. a + 0 == a
 * 2. a * 1 == a
 */
template <typename T>
concept FieldLike =
    std::floating_point<T> || std::same_as<std::complex<scalar_type_t<T>>, T>;

/// @brief concept definition for vector like
template <class Vec>
concept VectorLike = requires {
  FieldLike<typename vec_traits<Vec>::value_type>;
  { vec_traits<Vec>::SIZE } -> std::convertible_to<size_t>;
};

/// @brief vector vector compatability
template <class VecA, class VecB>
concept VecVecCompatible = VectorLike<VecA> && VectorLike<VecB> && requires {
  std::same_as<vec_type_t<VecA>, vec_type_t<VecB>> &&
      vec_size_v<VecA> == vec_size_v<VecB>;
};

/// @brief concept definition for matrix like
template <class Mat>
concept MatrixLike = requires {
  FieldLike<typename mat_traits<Mat>::value_type>;
  { mat_traits<Mat>::ROWS } -> std::convertible_to<size_t>;
  { mat_traits<Mat>::COLS } -> std::convertible_to<size_t>;
};

/// @brief concept square matrix like
template <class Mat>
concept SquareMatrixLike =
    MatrixLike<Mat> && mat_rows_v<Mat> == mat_cols_v<Mat>;

/// @brief concept for matrix vector mul compatibility
template <class Mat, class Vec>
concept MatVecCompatible = MatrixLike<Mat> && VectorLike<Vec> &&
                           std::same_as<mat_type_t<Mat>, vec_type_t<Vec>> &&
                           mat_cols_v<Mat> == vec_size_v<Vec>;

/// @brief concept for matrix matrix compatibility
template <class MatA, class MatB>
concept MatMulCompatible = MatrixLike<MatA> && MatrixLike<MatB> &&
                           std::same_as<mat_type_t<MatA>, mat_type_t<MatB>> &&
                           mat_cols_v<MatA> == mat_rows_v<MatB>;

}  // namespace linalg
