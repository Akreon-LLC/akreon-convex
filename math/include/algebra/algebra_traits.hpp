/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <type_traits>

namespace linalg {

/// @brief remove cv ref from type
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

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

  template <typename... Ts>
  static constexpr auto operator()(Ts&&... xs) {
    static_assert(sizeof...(xs) == N);
    Vector<T, N> V;
    int i = 0;
    auto set_elem = [&](const T& x) { V[i++] = x; };
    ((void)set_elem(std::forward<Ts>(xs)), ...);
    return V;
  }

  static constexpr auto fill(const T& x) {
    Vector<T, N> V;
    for (size_t i = 0; i < N; ++i) V[i] = x;
    return V;
  }
};

/// @brief helper for size of vector
template <class Vec>
inline constexpr size_t vec_size_v = vec_traits<remove_cvref_t<Vec>>::SIZE;

/// @brief alias for underlying type for vector
template <class Vec>
using vec_type_t = typename vec_traits<remove_cvref_t<Vec>>::value_type;

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
  static constexpr auto operator()(Ts&&... xs) {
    static_assert(sizeof...(xs) == Rows * Cols);
    Matrix<T, Rows, Cols> M;
    int i = 0;
    auto set_elem = [&](const T& x) {
      const auto col = i % Cols;
      const auto row = i / Cols;
      M[row][col] = x;
      i++;
    };
    ((void)set_elem(std::forward<Ts>(xs)), ...);
    return M;
  }

  static constexpr auto fill(const T& x) {
    Matrix<T, Rows, Cols> M;
    for (size_t i = 0; i < Rows; ++i)
      for (size_t j = 0; j < Cols; ++j) M[i][j] = x;
    return M;
  }
};

/// @brief alias for matrix type
template <class Mat>
using mat_type_t = typename mat_traits<remove_cvref_t<Mat>>::value_type;

/// @brief helper for size of matrix rows
template <class Mat>
inline constexpr size_t mat_rows_v = mat_traits<remove_cvref_t<Mat>>::ROWS;

/// @brief helper for size of matrix cols
template <class Mat>
inline constexpr size_t mat_cols_v = mat_traits<remove_cvref_t<Mat>>::COLS;

/// @brief base type traits definition for vector
template <class Vec>
struct space_traits;

template <typename T>
struct space_traits<Vector<Vector<T, 1>, 1>> {
  using value_type = T;
  using Point = Vector<T, 1>;

  static constexpr size_t DIM = 1;
};

/// @brief type traits definition for vector space (col vector)
template <typename T, size_t N>
struct space_traits<Vector<Vector<T, N>, 1>> {
  using value_type = T;
  using Point = Vector<T, N>;

  // (dakre) the dimension is the total number of independent scalar values
  // needed to define any vector in the space
  static constexpr size_t DIM = N;
};

/// @brief type traits definition for vector space (row vector)
template <typename T, size_t N>
struct space_traits<Vector<Vector<T, 1>, N>> {
  using value_type = T;
  using Point = Vector<T, N>;

  // (dakre) the dimension is the total number of independent scalar values
  // needed to define any vector in the space
  static constexpr size_t DIM = N;
};

/// @brief type traits definition for vector space
template <typename T, size_t M, size_t N>
  requires(M > 1 && N > 1)
struct space_traits<Vector<Vector<T, N>, M>> {
  using value_type = T;
  using Point = Vector<T, N>;

  // (dakre) the dimension is the total number of independent scalar values
  // needed to define any vector in the space
  static constexpr size_t DIM = M * N;
};

/// @brief alias for space type
template <class Space>
using space_type_t = typename space_traits<remove_cvref_t<Space>>::value_type;

/// @brief alias for point type in vector space
template <class Space>
using space_point_t = typename space_traits<remove_cvref_t<Space>>::Point;

/// @brief for dimension of vector space
template <class Space>
inline constexpr size_t space_dim_v = space_traits<remove_cvref_t<Space>>::DIM;

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

/// @brief alias for deducing vector of doubles with N elements
template <size_t N>
using Vectord = Vector<double, N>;

/// @brief alias for deducing vector of floats with N elements
template <size_t N>
using Vectorf = Vector<float, N>;

/// @brief alias for deducing vector of complex doubles with N elements
template <size_t N>
using Vectorcd = Vector<std::complex<double>, N>;

/// @brief alias for deducing vector of complex floats with N elements
template <size_t N>
using Vectorcf = Vector<std::complex<float>, N>;

/// @brief alias for deducing matrix as vector of vectors of type double with
/// Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixd = Vector<Vectord<Cols>, Rows>;

/// @brief alias for deducing matrix as vector of vectors of type float with
/// Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixf = Vector<Vectorf<Cols>, Rows>;

/// @brief alias for deducing matrix as vector of vectors of type complex double
/// with Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixcd = Vector<Vectorcd<Cols>, Rows>;

/// @brief alias for deducing matrix as vector of vectors of type complex float
/// with Rows x Cols elements
template <size_t Rows, size_t Cols>
using Matrixcf = Vector<Vectorcf<Cols>, Rows>;

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
concept VectorLike =
    (FieldLike<vec_type_t<Vec>> || FieldLike<mat_type_t<Vec>>) && requires {
      { vec_size_v<Vec> } -> std::convertible_to<size_t>;
    };

/// @brief vector vector compatability
template <class VecA, class VecB>
concept VecVecCompatible = VectorLike<VecA> && VectorLike<VecB> &&
                           std::same_as<vec_type_t<VecA>, vec_type_t<VecB>> &&
                           vec_size_v<VecA> == vec_size_v<VecB>;

/// @brief vector space concept
template <class Space>
concept VectorSpace =
    VectorLike<Space> &&
    requires(const space_point_t<Space>& x, const space_point_t<Space>& y,
             const space_type_t<Space>& alpha) {
      { x + y } -> std::convertible_to<space_point_t<Space>>;
      { x - y } -> std::convertible_to<space_point_t<Space>>;
      { alpha * x } -> std::convertible_to<space_point_t<Space>>;
      { alpha * (x + y) } -> std::convertible_to<space_point_t<Space>>;
      { alpha * x + alpha * y } -> std::convertible_to<space_point_t<Space>>;
      {
        space_type_t<Space>(1) * x
      } -> std::convertible_to<space_point_t<Space>>;
    };

/// @brief R^n space
template <class Space>
concept RnSpace = VectorSpace<Space> && !is_complex_v<space_type_t<Space>>;

/// @brief concept definition for matrix like
template <class Mat>
concept MatrixLike = FieldLike<mat_type_t<Mat>> && requires {
  { mat_rows_v<Mat> } -> std::convertible_to<size_t>;
  { mat_cols_v<Mat> } -> std::convertible_to<size_t>;
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
