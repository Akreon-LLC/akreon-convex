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

template <typename T, size_t N, typename Norm>
  requires(FieldLike<T>)
constexpr auto normalize(const Vector<T, N>& X, const Norm& norm) {
  const auto norm_val = norm(X);
  assert(norm_val > 0);
  return (1 / norm_val) * X;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto euclidean_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L2(X - Y);
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto manhattan_distance(const Vector<T, N>& X,
                                  const Vector<T, N>& Y) {
  return L1(X - Y);
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool all_equal(const algebra::Matrix<T, Rows, Cols>& A,
                         const algebra::Matrix<T, Rows, Cols>& B) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i)
    rv &= euclidean_distance(A(i), B(i)) <= 1e-10;
  return rv;
}

template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
constexpr bool rows_equal_cols(const Matrix<T, Rows, Cols>& A,
                               const Matrix<T, Cols, Rows>& B) {
  bool rv = true;
  for (size_t i = 0; i < Rows; ++i)
    rv &= euclidean_distance(A(i), get_col(B, i)) <= 1e-10;
  return rv;
}

template <typename T, size_t N>
  requires(FieldLike<T> && !is_complex_v<T>)
constexpr bool is_symmetric(const Matrix<T, N, N>& A) {
  return all_equal(A, transpose(A));
}

template <typename T, size_t N>
  requires(FieldLike<T> && is_complex_v<T>)
constexpr bool is_hermitian(const Matrix<T, N, N>& A) {
  return all_equal(A, conjugate_transpose(A));
}

template <typename T, size_t Rows, size_t Cols>
  requires(Rows > 0 && FieldLike<T>)
class QR {
 public:
  constexpr QR(const Matrix<T, Rows, Cols>& A) {
    // (dakre) TODO below is gram-schmidt in the future update to be
    // householder (improved accuracy)
    solve_gram_schmidt(A);
  }

  constexpr auto get_Q() const { return Q; }

  constexpr auto get_R() const { return R; }

  // A x = b => Q R x = b => R x = Q b^T
  constexpr auto solve(const Vector<T, Rows>& b) const {
    Vector<T, Cols> x, c;

    if constexpr (is_complex_v<T>)
      c = conjugate_transpose(Q) * b;
    else
      c = transpose(Q) * b;

    for (int i = static_cast<int>(Cols) - 1; i >= 0; --i) {
      T sum = 0;
      for (size_t j = i + 1; j < Cols; ++j) sum += (R(i)(j) * x(j));
      x[i] = (c(i) - sum) / R(i)(i);
    }

    return x;
  }

 private:
  constexpr void solve_gram_schmidt(const Matrix<T, Rows, Cols>& A) {
    // 1. edge case if A is all zeros current GS algo does not handle this well
    // so return I and zeros mat
    if (all_equal(A, Matrix<T, Rows, Cols>{})) {
      for (size_t i = 0; i < Cols; ++i) Q[i][i] = T(1);
      return;
    }

    // 2. gram schmidt algorithm below
    for (size_t i = 0; i < Cols; ++i) {
      auto v = get_col(A, i);
      for (size_t j = 0; j < i; ++j) {
        if constexpr (!is_complex_v<T>)
          R[j][i] = dot(get_col(Q, j), v);
        else
          R[j][i] = dot(conjugate(get_col(Q, j)), v);
        v = v - R[j][i] * get_col(Q, j);
      }
      const auto q_i = normalize(v, L2<T, Rows>);
      for (size_t j = 0; j < Rows; ++j) Q[j][i] = q_i(j);
      if constexpr (!is_complex_v<T>)
        R[i][i] = dot(q_i, get_col(A, i));
      else
        R[i][i] = dot(conjugate(q_i), get_col(A, i));
    }
  }

  Matrix<T, Rows, Cols> Q;
  Matrix<T, Cols, Cols> R;
};

template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && FieldLike<T>)
class EigenDecomposition {
 public:
  constexpr EigenDecomposition(const Matrix<T, Rows, Cols>& A,
                               scalar_type_t<T> tol = 1e-10) {
    solve_qr(A, tol);
  }

  constexpr auto get_eigenvalues() const { return get_diag(Lambda); }

  constexpr auto get_eigenvectors() const { return VT; }

 private:
  constexpr auto solve_qr(const Matrix<T, Rows, Cols>& A,
                          const scalar_type_t<T>& tol) {
    Lambda = A;
    VT = make_identity<T, Cols>();

    // Eigendecomposition works with diagonalizing the matrix to solve A X =
    // X \Lambda (\Lambda == eigenvalue matrix which is diagonal), the resulting
    // X is our eigenvector associated with the eigenvalues
    // 1. A_0 = A
    // 2. A_k = Q_k R_k
    // 3. A_{k+1} = R_k Q_k \equiv Q_k^T A_k Q_k similarity transform
    for (size_t i = 0; i < MAX_STACK_DEPTH; ++i) {
      const auto qr = QR(Lambda);
      const auto Q = qr.get_Q();
      const auto R = qr.get_R();
      Lambda = R * Q;
      VT = VT * Q;
      const auto off_diag = Lambda - make_diag(get_diag(Lambda));
      if (frobenius_norm(off_diag) < tol) break;
    }
  }

  Matrix<T, Rows, Cols> Lambda;
  Matrix<T, Cols, Cols> VT;
};

// A = U \Sigma V^T
template <typename T, size_t Rows, size_t Cols>
  requires(FieldLike<T>)
class SVD {
 public:
  constexpr SVD(const Matrix<T, Rows, Cols>& A, scalar_type_t<T> tol = 1e-10) {
    svd(A, tol);
  }

  constexpr auto get_left_singular_vectors() const { return U; }

  constexpr auto get_right_singular_vectors() const { return VT; }

  constexpr auto get_singular_values() const { return sigmas; }

 private:
  constexpr auto svd(const Matrix<T, Rows, Cols>& A,
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
    // resembling the right singular vectors space row space of Lambda above U
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

  Matrix<T, Cols, Cols> VT;
  Matrix<T, Rows, Rows> U;
  Vector<scalar_type_t<T>, Cols> sigmas;
};

}  // namespace algebra
