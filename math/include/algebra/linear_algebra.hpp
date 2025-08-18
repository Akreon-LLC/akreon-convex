/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "hessenberg.hpp"
#include "householder.hpp"
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
   * @brief get Q from compact householder Q
   *
   * To acquire Q we simply apply reflections in the reverse order with Q = I to
   * start. Then A = QR.
   *
   * @return orthonormal Q
   */
  constexpr auto get_Q() const { return get_unitary_matrix(); }

  /**
   * @brief basic get function for R matrix
   *
   * @return R
   */
  constexpr auto get_R() const { return get_upper_right_triangular_matrix(H); }

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

    const auto Q = get_Q();
    const auto R = get_R();

    if constexpr (is_complex_v<T>)
      c = conjugate_transpose(Q) * b;
    else
      c = transpose(Q) * b;

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
  constexpr void solve_householder(Matrix<T, Rows, Cols> A) {
    H = A;
    constexpr_for<0, Cols>([&]<size_t j>(std::integral_constant<size_t, j>) {
      const auto& Q_jm = get_n_rows<j, Rows>(H);
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
        for (size_t l = j, b = 0; l < Cols; ++l, ++b) H[k][l] = Q_jm_jn(a)(b);

      for (size_t k = j + 1, a = 1; k < Rows; ++k, ++a) H[k][j] = v(a)(0);
    });
  }

  /**
   * @brief get unitary matrix U from factored householder matrix
   *
   * To acquire U we simply apply reflections in the reverse order starting with
   * U = I => U^* A U = H
   *
   * @return unitary matrix U
   */
  constexpr auto get_unitary_matrix() const {
    auto U = make_identity<T, Rows>();
    constexpr_for<0, Cols>([&]<size_t jj>(std::integral_constant<size_t, jj>) {
      const size_t j = Cols - jj - 1;
      Vector<T, Rows - j> v;
      v[0] = T(1);
      for (size_t i = j + 1, a = 1; i < Rows; ++i, ++a) v[a] = H(i)(j);

      const auto& v_col = as_col_vector(v);

      auto Q_block = get_n_rows<j, Rows>(U);
      if constexpr (is_complex_v<T>)
        Q_block = Q_block -
                  (beta(j) * v_col) * (conjugate_transpose(v_col) * Q_block);
      else
        Q_block = Q_block - (beta(j) * v_col) * (transpose(v_col) * Q_block);

      for (size_t r = j, a = 0; r < Rows; ++r, ++a)
        for (size_t c = 0; c < Rows; ++c) U[r][c] = Q_block(a)(c);
    }); 
    return U;
  }

  /** @brief QR decomposition for matrix input A */
  Matrix<T, Rows, Cols> H;  ///< orthonormal basis Q (slighty different than GS
                            ///< as its fully compact and can use straight up)
  Vector<T, Rows> beta;     ///< beta scalars
};

/**
 * @brief EigenDecomposition class definition
 *
 * Solves the following equation A X = \Lambda X via QR and successive
 * similarity transforms to diagonalize A into eigenvalues.
 */
template <typename T, size_t N>
  requires(FieldLike<T>)
class EigenDecomposition {
 public:
  // @brief aliasing complex and real types for eigen decomposition
  using Real = scalar_type_t<T>;
  using Complex = std::conditional_t<is_complex_v<T>, T, std::complex<Real>>;

  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   */
  constexpr EigenDecomposition(const Matrix<T, N, N>& A,
                               scalar_type_t<T> tol = 1e-10)
      : converged(false) {
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

  /**
   * @brief basic get for convergence flag
   *
   * @return T/F
   */
  constexpr auto get_converged() const { return converged; }

 private:
  /**
   * @brief solve A X = \Lambda X via QR
   *
   * Following the practical QR Algorithm 7.5 in Matrix Computations
   *
   * @param matrix A of type T with N x N elements
   * @param tolerance dictating algorithm convergence
   *
   * @return (void)
   */
  constexpr void solve_qr(const Matrix<T, N, N>& A,
                          const scalar_type_t<T>& tol) {
    // 1. calculate hessenberg matrix
    const auto H_mat = HessenbergMatrix(A);
    auto H = H_mat();

    // 2. perform shifted QR iteration on H
    const auto I = make_identity<T, N>();
    // VT = I;

    // Eigendecomposition works with diagonalizing the matrix to solve A X =
    // X \Lambda (\Lambda == eigenvalue matrix which is diagonal)
    for (size_t i = 0; i < MAX_STEP_EVAL; ++i) {
      const auto mu = H(N - 1)(N - 1);
      H = H - mu * I;
      auto qr = QR(H);
      auto U = qr.get_Q();
      auto R = qr.get_R();

      // R U + mu I = U^T (U R + mu I) U = U^T H U (similarity transform)
      H = R * U + mu * I;

      // (dakre) TODO fix eigenvectors
      // VT = VT * U;

      // (dakre) iterate through the upper hessenberg to acquire eigenvalues
      if constexpr (is_complex_v<T>) {
        for (size_t j = 0; j < N; ++j) lambdas[j] = H(j)(j);
      } else {
        // (dakre) Hessenberg forms Real Schur Decomposition on the
        // quasi-triangular matrix. Complex eigs come in pairs, and if c is zero
        // then its strictly real 1x1 block
        for (size_t j = 0; j < N; ++j) {
          Real a = H(j)(j);
          if (j + 1 == N) {
            lambdas[j] = Complex(a, 0);
            continue;
          }
          Real c = H(j + 1)(j);
          if (constexpr_abs(c) < tol) {
            lambdas[j] = Complex(a, 0);
            continue;
          }
          Real d = H(j + 1)(j + 1);
          Real b = H(j)(j + 1);
          lambdas[j] = Complex(a, b);
          lambdas[j + 1] = Complex(c, d);
          j++;
        }
      }

      scalar_type_t<T> res = scalar_type_t<T>(0);
      for (size_t j = 1; j < N; ++j) {
        for (size_t k = 0; k < std::min(j, N); ++k) {
          if constexpr (is_complex_v<T>)
            res += (std::conj(H(j)(k)) * H(j)(k)).real();
          else
            res += H(j)(k) * H(j)(k);
        }
      }

      if (res < tol * tol) {
        converged = true;
        break;
      }
    }
  }

  /** @brief eigendecomposition member variables */
  Vector<Complex, N> lambdas;  ///< eigenvalues
  Matrix<Complex, N, N> VT;    ///< eigenvectors
  bool converged;
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
