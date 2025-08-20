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

  // (dakre) TODO handle this a little better in the future
  static constexpr size_t n_steps = 64;
  static constexpr scalar_type_t<T> tol = 1e-10;

  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   */
  constexpr EigenDecomposition(const Matrix<T, N, N>& A) : converged(false) {
    solve_qr(A);
  }

  /**
   * @brief basic get eigenvalues function
   *
   * @return vector of eigenvalues
   */
  constexpr auto get_eigenvalues() const { return lambda; }

  /**
   * @brief basic get eigenvectors function
   *
   * @return matrix of eigenvectors
   */
  constexpr auto get_eigenvectors() const { return V; }

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
  constexpr void solve_qr(const Matrix<T, N, N>& A) {
    // 1. calculate hessenberg matrix
    const auto H_mat = HessenbergMatrix(A);
    auto H = H_mat();

    // 2. setup unitary basis for eigenvectors
    auto U = H_mat.get_U();

    // 3. the practical QR iteration on H -- form quasi triangular matrix in
    // real schur form and extract eigenvalues depending on real / complex form
    // of 2x2 or 1x1 schur blocks
    for (size_t i = 0; i < n_steps; ++i) {
      auto qr = QR(H);
      auto Q = qr.get_Q();
      auto R = qr.get_R();

      // R U = U^T (U R) U = U^T H U (similarity transform)
      H = R * Q;

      // continue to map orthogonal basis Q onto unitary coords from orig H
      U = U * Q;

      // (dakre) iterate through the upper hessenberg to acquire eigenvalues
      if constexpr (is_complex_v<T>) {
        for (size_t j = 0; j < N; ++j) lambda[j] = H(j)(j);
      } else {
        // (dakre) Hessenberg forms Real Schur Decomposition on the
        // quasi-triangular matrix. Complex eigs come in pairs, and if c is zero
        // then its strictly real 1x1 block
        for (size_t j = 0; j < N; ++j) {
          const Real& a = H(j)(j);
          if (j + 1 == N) {
            lambda[j] = Complex(a, 0);
            continue;
          }
          const Real& c = H(j + 1)(j);
          const Real& d = H(j + 1)(j + 1);
          if (constexpr_abs(c) < tol * (constexpr_abs(c) + constexpr_abs(d))) {
            lambda[j] = Complex(a, 0);
            continue;
          }

          // (dakre) \lambda^2 - (a + d) \lambda + (ad - bc) = 0
          // t = a + d, \delta = ad - bc, \mu = 0.5 * (a - d)
          // t^2 - 4 \delta then sub in the scalars above to form \lambda
          const Real& b = H(j)(j + 1);
          const auto t = a + d;
          const auto mu = 0.5 * (a - d);
          const auto D = mu * mu + b * c;

          if (D >= 0) {
            lambda[j] = Complex(t / 2 + constexpr_sqrt(D), 0.0);
            lambda[j + 1] = Complex(t / 2 - constexpr_sqrt(D), 0.0);
          } else {
            lambda[j] = Complex(t / 2, constexpr_sqrt(D));
            lambda[j + 1] = Complex(t / 2, -1 * constexpr_sqrt(D));
          }

          j++;
        }
      }

      scalar_type_t<T> res = scalar_type_t<T>(0);
      for (size_t r = 2; r < N; r += 2) {
        for (size_t c = 0; c < r - 1; ++c) {
          if constexpr (is_complex_v<T>)
            res += (std::conj(H(r)(c)) * H(r)(c)).real();
          else
            res += H(r)(c) * H(r)(c);
        }
      }

      if (N > 2 && res < tol * tol) {
        converged = true;
        break;
      } else {
        // (dakre) TODO this is the 2x2 case, let it run for fixed n steps for
        // now & possibly think about better way to handle in the future
        converged = true;
      }
    }

    // 4. back substitute now for solving eigenvectors
    // a. (H - \lambda I) x = 0
    // b. apply coordinate map to x from unitary basis U to get it aligned in A
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
      Vector<Complex, N> x;
      if (!is_real(lambda[i]) && i > 0) {
        seed_complex_block(H, i, x);

        back_sub_1x1_block(H, i, x);

        auto v = map_to_A(U, x);
        for (size_t j = 0; j < N; ++j) V[j][i] = v(j);

        i--;

        v = map_to_A(U, conjugate(x));
        for (size_t j = 0; j < N; ++j) V[j][i] = v(j);
      } else {
        x[i] = Complex(1);

        back_sub_1x1_block(H, i, x);

        auto v = map_to_A(U, x);
        for (size_t j = 0; j < N; ++j) V[j][i] = v(j);
      }
    }
  }

  /**
   * @brief back substitution for the real 1x1 block from Schur Form
   *
   * @param real schur upper triangular matrix H
   * @param index of eigenvalue to back solve
   * @param mutable eigenvector x
   *
   * @return (void)
   */
  constexpr void back_sub_1x1_block(const Matrix<T, N, N>& H, const int& idx,
                                    Vector<Complex, N>& x) {
    // (dakre) solve (H - \lambda I) x = 0 by way of back sub from idx to top
    // (H_rr - \Lambda) x_r + \sum_0^idx H_rc * x_c = 0 ... isolate x_r and
    // solve
    for (int r = idx - 1; r >= 0; --r) {
      Complex sum(0);
      for (int c = r + 1; c <= idx; ++c) sum += Complex(H(r)(c)) * x[c];
      x[r] = -sum / (Complex(H(r)(r)) - lambda(idx));
    }
  }

  /**
   * @brief seed the complex 2x2 block from Schur Form
   *
   * @param real schur upper triangular matrix H
   * @param index of eigenvalue to back solve
   * @param mutable eigenvector x
   *
   * @return (void)
   */
  constexpr void seed_complex_block(const Matrix<T, N, N>& H, const int& idx,
                                    Vector<Complex, N>& x) {
    const auto& a = H(idx - 1)(idx - 1);
    const auto& b = H(idx - 1)(idx);
    const auto& c = H(idx)(idx - 1);
    const auto& d = H(idx)(idx);

    // seed eigenvector at position idx to kick of back substitution
    const Complex la = lambda[idx] - Complex(a);
    Complex z0, z1;
    if (L2<T, 1>(Vector<T, 1>{b}) >= L2<Complex, 1>(la)) {
      // Use [b; lambda - a]
      z0 = Complex(b);
      z1 = la;
    } else {
      // Use [lambda - d; c]
      z0 = lambda[idx] - Complex(d);
      z1 = Complex(c);
    }
    Real s = constexpr_sqrt(L2<Complex, 1>(z0) + L2<Complex, 1>(z1));
    if (s > Real(0)) {
      z0 /= s;
      z1 /= s;
    }

    x[idx - 1] = z0;
    x[idx] = z1;
  }

  /**
   * @brief map eigenvector from H to A
   *
   * @param unitary basis U from real schur upper triangular form on H
   * @param eigenvector v
   *
   * @return eigenvector in basis A
   */
  constexpr auto map_to_A(const Matrix<T, N, N>& U,
                          const Vector<Complex, N>& x) {
    Vector<Complex, N> v;

    for (size_t j = 0; j < N; ++j) {
      Complex sum(0);
      for (size_t k = 0; k < N; ++k) sum += Complex(U(j)(k)) * x(k);
      v[j] = sum;
    }

    return normalize(v, L2<Complex, N>);
  }

  /** @brief eigendecomposition member variables */
  Vector<Complex, N> lambda;  ///< eigenvalues
  Matrix<Complex, N, N> V;    ///< eigenvectors
  bool converged;             ///< eigenvalue convergence flag
};

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
  // @brief aliasing complex and real types for eigen decomposition
  using Real = scalar_type_t<T>;
  using Complex = std::conditional_t<is_complex_v<T>, T, std::complex<Real>>;

  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   */
  constexpr SVD(const Matrix<T, Rows, Cols>& A) : converged(false) { svd(A); }
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
  constexpr auto get_singular_values() const { return S; }

  /**
   * @brief basic get for convergence flag
   *
   * @return T/F
   */
  constexpr auto get_converged() const { return converged; }

 private:
  /**
   * @brief solve A = U \Sigma V^T
   *
   * @param matrix A of type T with Rows x Cols elements
   *
   * @return (void)
   */
  constexpr void svd(const Matrix<T, Rows, Cols>& A) {
    // 1. compute A^T A then solve eigen decomposition
    Matrix<T, Cols, Cols> M;
    if constexpr (!is_complex_v<T>)
      M = transpose(A) * A;
    else
      M = conjugate_transpose(A) * A;

    const auto result = EigenDecomposition(M);
    converged = result.get_converged();

    // 2. convert eigenvalues to singular values
    const auto tmp_lambdas = result.get_eigenvalues();
    Vector<Real, Cols> lambdas;
    for (size_t i = 0; i < Cols; ++i) {
      assert(is_real(tmp_lambdas(i)));
      lambdas[i] = tmp_lambdas(i).real();
    }

    // (dakre) these eigenvectors are projected onto the row space thus
    // resembling the right singular vectors of the row space of Lambda above
    // U (A^T A) VT -> U \Sigma VT
    auto V = result.get_eigenvectors();

    for (size_t i = 0; i < Cols; ++i)
      S[i][i] = constexpr_sqrt(std::max<Real>(0, lambdas(i)));

    // 3. back-solve for U with VT & sigmas
    // \Sigma^{-1} A V = U
    U = make_identity<Complex, Rows>();

    for (size_t i = 0; i < Cols; ++i) {
      const auto u_i = (1 / S(i)(i)) * as_complex_matrix(A) * get_col(V, i);
      for (size_t j = 0; j < Rows; ++j) U[j][i] = u_i(j);
    }

    VT = conjugate_transpose(V);

    // 4. TODO sort singular values & vectors largest -> smallest
  }

  /** @brief SVD member variables */
  Matrix<Complex, Cols, Cols> VT;  ///< right singular vectors
  Matrix<Complex, Rows, Rows> U;   ///< left singular vectors
  Matrix<Real, Rows, Cols> S;      ///< singular values
  bool converged;                  ///< converged
};

}  // namespace algebra
