/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "matrix_norms.hpp"
#include "vector_norms.hpp"

/* (dakre) Global TODO
 * 1. punting on cholesky and LDL right now, with current approach and checking
 * positive definiteness im not seeing the performance boost needed for MVP,
 * perhaps later on integrating it would make sense
 */

namespace algebra {

/**
 * @brief Normalize a vector given a norm function
 *
 * @param vector of type T with N elements
 * @param reference to a Norm object (L1, L2, max_norm, etc.)
 *
 * @return normalized vector
 */
template <typename T, size_t N, typename Norm>
  requires(FieldLike<T>)
constexpr auto normalize(const Vector<T, N>& X, const Norm& norm) {
  const auto norm_val = norm(X);
  assert(norm_val > 0);
  return (1 / norm_val) * X;
}

/**
 * @brief Compute the euclidean distance between two vectors (L2 norm of delta)
 *
 * @param vector X of type T with N elements
 * @param vector Y of type T with N elements
 *
 * @return distance between two vectors as a scalar
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto euclidean_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L2(X - Y);
}

/**
 * @brief Compute the manhattan distance between two vectors (L1 norm of delta)
 *
 * @param vector X of type T with N elements
 * @param vector Y of type T with N elements
 *
 * @return distance between two vectors as a scalar
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto manhattan_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L1(X - Y);
}

/**
 * @brief checks if two given matrix are equivalent
 *
 * @param matrix A of type T with Rows x Cols elements
 * @param vector B of type T with Rows x Cols elements
 * @param tolerance set to 1e-10 by default
 *
 * @return T/F
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool all_equal(const algebra::Matrix<T, Rows, Cols>& A,
                         const algebra::Matrix<T, Rows, Cols>& B,
                         scalar_type_t<T> tol = 1e-10) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i) rv &= euclidean_distance(A(i), B(i)) <= tol;
  return rv;
}

/**
 * @brief checks if the row space of one matrix is equivalent to the column
 * space of another matrix
 *
 * @param matrix A of type T with Rows x Cols elements
 * @param vector B of type T with Cols x Rows elements
 *
 * @return T/F
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool rows_equal_cols(const Matrix<T, Rows, Cols>& A,
                               const Matrix<T, Cols, Rows>& B) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i)
    rv &= euclidean_distance(A(i), get_col(B, i)) <= 1e-10;
  return rv;
}

/**
 * @brief checks if a given matrix is symmetric
 *
 * @param matrix A of type T with N x N elements
 *
 * @return T/F
 */
template <typename T, size_t N>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr bool is_symmetric(const Matrix<T, N, N>& A) {
  return all_equal(A, transpose(A));
}

/**
 * @brief checks if a given matrix is hermitian
 *
 * @param matrix A of type T with N x N elements
 * @return T/F
 */
template <typename T, size_t N>
  requires(FieldLike<T> && is_complex_v<T>)
constexpr bool is_hermitian(const Matrix<T, N, N>& A) {
  return all_equal(A, conjugate_transpose(A));
}

/**
 * @brief QR decomposition class definition
 *
 * Orthonormalizes a given Matrix into an orthogonal basis Q and upper right
 * triangular matrix R. Deprecated gram schmidt for householder for better
 * numerical properties.
 *
 * Q is now compact in householder where A = QR => Q = A R^{-1} in the
 * representation of this class
 *
 * For short and fat matrices just transpose A -> QR -> transpose result back
 * into A = L Q result
 */
template <typename T, size_t Rows, size_t Cols>
  requires(Rows >= Cols && Rows > 0 && FieldLike<T>)
class QR {
 public:
  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   */
  constexpr QR(const Matrix<T, Rows, Cols>& A) { solve_householder(A); }

  /**
   * @brief basic get function for Q matrix
   *
   * @return Q
   */
  constexpr auto get_householder_Q() const { return Q; }

  /**
   * @brief get Q from compact householder Q
   *
   * To acquire Q we simply apply reflections in the reverse order with Q = I to
   * start. Then A = QR.
   *
   * @return orthonormal Q
   */
  constexpr auto get_Q() const {
    const size_t k = std::min(Rows, Cols);
    auto Q_bar = make_identity<T, Rows>();
    constexpr_for<0, k>([&]<size_t jj>(std::integral_constant<size_t, jj>) {
      const size_t j = k - jj - 1;
      Vector<T, Rows - j> v;
      v[0] = T(1);
      for (size_t i = j + 1, a = 1; i < Rows; ++i, ++a) v[a] = Q(i)(j);

      auto v_col = as_col_vector(v);

      auto Q_block = get_n_rows<j, Rows>(Q_bar);
      if constexpr (is_complex_v<T>)
        Q_block = Q_block -
                  (beta(j) * v_col) * (conjugate_transpose(v_col) * Q_block);
      else
        Q_block = Q_block - (beta(j) * v_col) * (transpose(v_col) * Q_block);

      for (size_t r = j, a = 0; r < Rows; ++r, ++a)
        for (size_t c = 0; c < Rows; ++c) Q_bar[r][c] = Q_block(a)(c);
    });
    return Q_bar;
  }

  /**
   * @brief basic get function for R matrix
   *
   * @return R
   */
  constexpr auto get_R() const { return get_upper_right_triangular_matrix(Q); }

  /**
   * @brief get rank of matrix
   *
   * @return rank
   */
  constexpr auto get_rank(scalar_type_t<T> tol = 1e-6) const {
    const auto& R = get_R();
    const size_t mn = std::min(Rows, Cols);
    size_t rank = 0;
    for (size_t i = 0; i < mn; ++i)
      if (L2<T, 1>(Vector<T, 1>{R(i)(i)}) > tol) rank += 1;
    return rank;
  }

  /**
   * @brief least squares solver A x = b using QR decomposition
   *
   * LSQ method: A x = b => Q R x = b => R x = Q^T b
   *
   * @param vector b representing the outputs or observables of a linear system
   *
   * @return x
   */
  constexpr auto solve(const Vector<T, Rows>& b) const {
    Vector<T, Rows> c;
    Vector<T, Cols> x;

    const auto Q_bar = get_Q();
    const auto R = get_R();

    if constexpr (is_complex_v<T>)
      c = conjugate_transpose(Q_bar) * b;
    else
      c = transpose(Q_bar) * b;

    constexpr auto mn = std::min(Rows, Cols);
    for (int i = static_cast<int>(mn) - 1; i >= 0; --i) {
      T sum = 0;
      for (size_t j = i + 1; j < Cols; ++j) sum += (R(i)(j) * x(j));
      x[i] = (c(i) - sum) / R(i)(i);
    }

    return x;
  }

 private:
  /**
   * @brief construct householder matrix from A (solve QR where Q is compact
   * form)
   *
   * Premultiplication matrix-vector product update with rank-1 update
   * PA = (I - beta v v^T) A = A - (beta v)*(v^T A)
   *
   * Postmultiplication
   * AP = A*(I - beta v v^T) = A - (A v)(beta v)^T
   *
   * Householder essentially annihilates subdiagonal column entries based on
   * each successive iterations reflections. Look at premultiplication case:
   *
   * P0 A annhiliates first col and creates basis vector in spanning first dir.
   * P1 P0 A annihilates second col and creates basis spanning 2 dim now... so
   * on and so forth till R = Pk Pk-1 ... P0 A where R is our right upper
   * triangular matrix
   *
   * @param matrix A of type T with Rows x Cols elements
   *
   * @return (void)
   */
  constexpr void solve_householder(const Matrix<T, Rows, Cols>& A) {
    Q = A;
    constexpr auto mn = std::min(Rows, Cols);
    constexpr_for<0, mn>([&]<size_t j>(std::integral_constant<size_t, j>) {
      const auto& Q_jm = get_n_rows<j, Rows>(Q);
      const auto& Q_jm_j = get_col(Q_jm, j);

      const auto& q = HouseholderVector(Q_jm_j);
      beta[j] = q.get_beta();
      const auto& v = as_col_vector(q.get_v());

      auto Q_jm_jn = get_n_cols<j, Cols>(Q_jm);
      if constexpr (is_complex_v<T>)
        Q_jm_jn = Q_jm_jn - (beta[j] * v) * (conjugate_transpose(v) * Q_jm_jn);
      else
        Q_jm_jn = Q_jm_jn - (beta[j] * v) * (transpose(v) * Q_jm_jn);

      for (size_t k = j, a = 0; k < Rows; ++k, ++a)
        for (size_t l = j, b = 0; l < Cols; ++l, ++b) Q[k][l] = Q_jm_jn(a)(b);

      for (size_t k = j + 1, a = 1; k < Rows; ++k, ++a) Q[k][j] = v(a)(0);
    });
  }

  /** @brief QR decomposition for matrix input A */
  Matrix<T, Rows, Cols> Q;  ///< orthonormal basis Q (slighty different than GS
                            ///< as its fully compact and can use straight up)
  Vector<T, Rows> beta;     ///< beta scalars
};

/**
 * @brief EigenDecomposition class definition
 *
 * Solves the following equation A X = \Lambda X via QR and successive
 * similarity transforms to diagonalize A into eigenvalues.
 *
 * Following orthogonal iteration outlined in Matrix Computations 7.3.2 - 7.3.3
 * QR Iteration-- TODO there is speed up using givens and householder
 * reflections more directly but I will look at that later on
 */
template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && FieldLike<T>)
class EigenDecomposition {
 public:
  constexpr static size_t K = std::min(Rows, Cols);
  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   */
  constexpr EigenDecomposition(const Matrix<T, Rows, Cols>& A,
                               scalar_type_t<T> tol = 1e-10) {
    solve_qr(A, tol);
  }

  /**
   * @brief basic get eigenvalues function
   *
   * @return vector of eigenvalues
   */
  constexpr auto get_eigenvalues() const { return lambdas; }

  /**
   * @brief basic get eigenvectors function
   *
   * @return matrix of eigenvectors
   */
  constexpr auto get_eigenvectors() const { return VT; }

 private:
  /**
   * @brief solve A X = \Lambda X via QR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   *
   * @return (void)
   */
  constexpr void solve_qr(const Matrix<T, Rows, Cols>& A,
                          const scalar_type_t<T>& tol) {
    Matrix<T, Rows, Cols> Z = A;

    auto qr = QR(Z);
    auto Q = qr.get_Q();
    auto R = qr.get_R();

    VT = make_identity<T, Cols>();

    // Eigendecomposition works with diagonalizing the matrix to solve A X =
    // X \Lambda (\Lambda == eigenvalue matrix which is diagonal)
    for (size_t i = 0; i < MAX_STEP_EVAL; ++i) {
      Z = A * Q;
      qr = QR(Z);
      Q = qr.get_Q();
      R = qr.get_R();

      Matrix<T, Rows, Cols> Tk;
      if constexpr (is_complex_v<T>)
        Tk = conjugate_transpose(Q) * (A * Q);
      else
        Tk = transpose(Q) * (A * Q);

      VT = VT * Q;
      lambdas = get_diag(Tk);

      // (dakre) NOTE removing fro norm here as it creates too many steps for
      // compiler to evaluate nominally... if (frobenius_norm(off_diag) < tol)
      // break;
      scalar_type_t<T> res = scalar_type_t<T>(0);
      for (size_t i = 1; i < Rows; ++i) {
        for (size_t j = 0; j < std::min(i, Cols); ++j) {
          if constexpr (is_complex_v<T>)
            res += (std::conj(Tk(i)(j)) * Tk(i)(j)).real();
          else
            res += Tk(i)(j) * Tk(i)(j);
        }
      }

      if (res < tol * tol) break;
    }
  }

  /** @brief eigendecomposition member variables */
  Vector<T, K> lambdas;      ///< eigenvalues
  Matrix<T, Cols, Cols> VT;  ///< eigenvectors
};

#if 0
/**
 * @brief Singular value decomposition class definition
 *
 * Decompose any matrix A into U \Sigma V^T where U are the left singular
 * vectors, \Sigma are the singular values, and V^T are the right singular
 * vectors
 */
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
class SVD {
 public:
  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   */
  constexpr SVD(const Matrix<T, Rows, Cols>& A, scalar_type_t<T> tol = 1e-10) {
    svd(A, tol);
  }
  /**
   * @brief basic get for left singular vectors
   *
   * @return matrix of left singular vectors
   */
  constexpr auto get_left_singular_vectors() const { return U; }

  /**
   * @brief basic get for right singular vectors
   *
   * @return matrix of right singular vectors
   */
  constexpr auto get_right_singular_vectors() const { return VT; }

  /**
   * @brief basic get for singular values
   *
   * @return matrix of singular values
   */
  constexpr auto get_singular_values() const { return sigmas; }

 private:
  /**
   * @brief solve A = U \Sigma V^T
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   *
   * @return (void)
   */
  constexpr void svd(const Matrix<T, Rows, Cols>& A,
                     const scalar_type_t<T>& tol = 1e-10) {
    Matrix<T, Cols, Cols> M;
    if constexpr (!is_complex_v<T>)
      M = transpose(A) * A;
    else
      M = conjugate_transpose(A) * A;

    const auto result = EigenDecomposition(M, tol);

    const auto tmp_lambdas = result.get_eigenvalues();
    Vector<scalar_type_t<T>, Cols> lambdas;
    if constexpr (!is_complex_v<T>)
      lambdas = tmp_lambdas;
    else {
      for (size_t i = 0; i < Cols; ++i) {
        assert(is_real(tmp_lambdas(i)));
        lambdas[i] = tmp_lambdas(i).real();
      }
    }

    // (dakre) these eigenvectors are projected onto the row space thus
    // resembling the right singular vectors of the row space of Lambda above U
    // (A^T A) VT -> U \Sigma VT
    VT = result.get_eigenvectors();

    for (size_t i = 0; i < Cols; ++i)
      sigmas[i] = constexpr_sqrt(std::max<scalar_type_t<T>>(0, lambdas(i)));

    // \Sigma^{-1} A V = U
    U = make_identity<T, Rows>();

    const auto V = transpose(VT);

    constexpr auto mn = std::min(Rows, Cols);
    for (size_t i = 0; i < mn; ++i) U[i] = (1 / sigmas(i)) * A * get_col(V, i);

    U = transpose(U);

    // (dakre) re-orthonormalize U and VT to ensure vectors form an orthonormal
    // basis
    const auto qr_U = QR(U);
    U = qr_U.get_Q();
    const auto qr_VT = QR(VT);
    VT = qr_VT.get_Q();
  }

  /** @brief SVD member variables */
  Matrix<T, Cols, Cols> VT;               ///< right singular vectors
  Matrix<T, Rows, Rows> U;                ///< left singular vectors
  Vector<scalar_type_t<T>, Cols> sigmas;  ///< singular values
};
#endif

}  // namespace algebra
