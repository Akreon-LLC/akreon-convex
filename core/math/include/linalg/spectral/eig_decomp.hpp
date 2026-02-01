/* Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include "linalg/operator/hessenberg.hpp"
#include "qr.hpp"

namespace linalg {

/**
 * @brief EigenDecomposition class definition
 *
 * Solves the following equation A X = \Lambda X via QR and successive
 * similarity transforms to diagonalize A into eigenvalues.
 */
template <SquareMatrixLike Mat>
struct EigenDecomposition {
  // @brief alias for underlying type
  using T = mat_type_t<Mat>;

  // @brief aliasing complex and real types for eigen decomposition
  using Real = scalar_type_t<T>;
  using Complex = std::conditional_t<ComplexLike<T>, T, std::complex<Real>>;

  static constexpr size_t STEPS = SolverTraits<Real>().steps;
  static constexpr Real TOL = SolverTraits<Real>().tol;
  static constexpr size_t N = mat_rows_v<Mat>;

  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   * @param tolerance dictating algorithm convergence
   */
  constexpr EigenDecomposition(const Mat& A) : converged(false) { solve_qr(A); }

  /**
   * @brief get spectral radius
   *
   * AKA maximum magnitude of a matrix as dictated by its max abs eigenvalue
   *
   * @return \rho
   */
  constexpr auto spectral_radius() const {
    Complex rho(0.0);
    for (size_t i = 0; i < N; ++i) {
      const auto mag_rho =
          constexpr_sqrt(rho.real() * rho.real() + rho.imag() * rho.imag());
      const auto mag_lambda =
          constexpr_sqrt(lambda(i).real() * lambda(i).real() +
                         lambda(i).imag() * lambda(i).imag());
      rho = mag_lambda > mag_rho ? lambda(i) : rho;
    }
    return rho;
  }

  /**
   * @brief get determinant
   *
   * \product \lambda_i = det(A)
   *
   * @return det (T)
   */
  constexpr auto determinant() const {
    Complex det(1);
    for (const auto& lam : lambda) det *= lam;
    return det;
  }

  /** @brief eigendecomposition member variables */
  Vector<Complex, N> lambda;  ///< eigenvalues
  Matrix<Complex, N, N> V;    ///< eigenvectors
  bool converged;             ///< eigenvalue convergence flag

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
  constexpr void solve_qr(const Mat& A) {
    // 1. calculate hessenberg matrix
    const auto H_mat = HessenbergMatrix(A);
    auto H = H_mat.H();

    // 2. setup unitary basis for eigenvectors
    auto U = H_mat.U();

    Vector<Complex, N> eigs;

    // 3. the practical QR iteration on H -- form quasi triangular matrix in
    // real schur form and extract eigenvalues depending on real / complex form
    // of 2x2 or 1x1 schur blocks
    for (size_t i = 0; i < STEPS; ++i) {
      const auto qr = QR(H);
      const auto Q = qr.Q();
      const auto R = qr.R();

      // R U = U^T (U R) U = U^T H U (similarity transform)
      H = R * Q;

      // continue to map orthogonal basis Q onto unitary coords from orig H
      U = U * Q;

      // (dakre) iterate through the upper hessenberg to acquire eigenvalues
      if constexpr (ComplexLike<T>) {
        for (size_t j = 0; j < N; ++j) eigs[j] = H(j)(j);
      } else {
        // (dakre) Hessenberg forms Real Schur Decomposition on the
        // quasi-triangular matrix. Complex eigs come in pairs, and if c is zero
        // then its strictly real 1x1 block
        for (size_t j = 0; j < N; ++j) {
          const Real& a = H(j)(j);
          if (j + 1 == N) {
            eigs[j] = Complex(a, 0);
            continue;
          }
          const Real& c = H(j + 1)(j);
          const Real& d = H(j + 1)(j + 1);
          if (constexpr_abs(c) < TOL * (constexpr_abs(c) + constexpr_abs(d))) {
            eigs[j] = Complex(a, 0);
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
            eigs[j] = Complex(t / 2 + constexpr_sqrt(D), 0.0);
            eigs[j + 1] = Complex(t / 2 - constexpr_sqrt(D), 0.0);
          } else {
            eigs[j] = Complex(t / 2, constexpr_sqrt(D));
            eigs[j + 1] = Complex(t / 2, -1 * constexpr_sqrt(D));
          }

          j++;
        }
      }

      if (N > 2) {
        Real res(0);
        for (size_t r = 2; r < N; r += 2) {
          for (size_t c = 0; c < r - 1; ++c) {
            if constexpr (ComplexLike<T>)
              res += (std::conj(H(r)(c)) * H(r)(c)).real();
            else
              res += H(r)(c) * H(r)(c);
          }
        }

        if (res < TOL * TOL) {
          converged = true;
          break;
        }
      } else {
        // (dakre) TODO N = 2 keeps schur block form so need to determine
        // appropriate way to say "converged"
        converged = true;
      }
    }

    // 3.a prepare back solve for eigenvectors via Jordan blocks
    std::array<bool, N> used{false};

    auto indices_equal = [&](const Vector<Complex, N>& D,
                             Complex val) -> std::array<int, N> {
      std::array<int, N> idx;
      idx.fill(-1);
      for (int j = 0; j < int(N); ++j)
        if (is_same(D(j), val)) idx[j] = j;
      return idx;
    };

    auto take_last_unused = [&](std::array<int, N>& slots) -> int {
      for (int j = 0; j < int(N); ++j) {
        if (!used[j] && slots[j] != -1) {
          used[j] = true;
          return j;
        }
      }
      return -1;
    };

    // 4. back substitute now for solving eigenvectors
    // a. check for normal form (AM == 1)
    //  - solve for root (H - \lambda I) x = 0
    //  - apply coordinate map to x (evec) from unitary basis U to get it
    //  aligned in A
    // b. solve jordan block (AM > 1)
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
      if (used[i]) continue;

      // Jordan block
      // - AM = algebraic multiplicity (# of repeated \lambda)
      // - GM = geometric multiplicity (# of linearly independent
      // eigenvectors for \lambda)
      // - Defect = # of extra generalized evecs to span the space the Jordan
      // block lies in
      int AM = 0;
      std::for_each(eigs.begin(), eigs.end(),
                    [&i, &eigs, &AM](const auto& val) {
                      AM += constexpr_absc(eigs(i) - val) < TOL;
                    });
      int GM = 0, defect = 0;
      if (AM > 1) {
        const auto I = make_identity<Complex, N>();
        GM = N - QR(as_complex_matrix(A) - eigs(i) * I).rank();
        defect = std::max(0, AM - GM);
      }

      if (0 == defect) {
        Vector<Complex, N> x;

        // (a) solve normally
        if (!is_real(eigs(i)) && i > 0) {
          // (0) seed & back solve homogenous case (H - \lambda I) x = 0
          seed_complex_block(H, i, x);
          back_sub_1x1_block(H, i, x);

          // (1) map vec(s)... conjugate pairs onto A
          auto v = map_to_A(U, x);
          for (size_t j = 0; j < N; ++j) V[j][i] = v(j);

          i--;

          v = map_to_A(U, conjugate(x));
          for (size_t j = 0; j < N; ++j) V[j][i] = v(j);

          // (2) set eig to lambda
          lambda[i] = eigs(i);

          used[i] = true;
        } else {
          // (0) seed & back solve homogenous case (H - \lambda I) x = 0
          seed_real_block(i, x);
          back_sub_1x1_block(H, i, x);

          // (1) map vec onto A
          auto v = map_to_A(U, x);
          for (size_t j = 0; j < N; ++j) V[j][i] = v(j);

          // (2) set eig to lambda
          lambda[i] = eigs(i);

          used[i] = true;
        }
      } else {
        // (b) solve for jordan block
        if (!is_real(eigs(i)) && i > 0) {
          // (0) setup jordan chain with padded dim to defect to ensure valid
          // block creation
          const int chain_len = std::min(1 + defect, (i + 1) / 2);
          Vector<Vector<Complex, N>, N> chain;

          // (1) seed and back solve for homogenous case (H - \lambda I) x = 0
          seed_complex_block(H, i, chain[0]);
          back_sub_1x1_block(H, i, chain[0]);

          // (2) back solve for each (H - \lambda I) x_k = x_{k+1}
          for (int k = 1; k < chain_len; ++k)
            back_sub_shifted_rhs(H, lambda(i), chain(k - 1), chain[k]);

          // (3) map back to A... [i, i-1, ...]
          auto slots = indices_equal(eigs, eigs(i));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots);
            if (slot_idx < 0) break;

            auto v = map_to_A(U, chain[k]);

            for (size_t j = 0; j < N; ++j) V[j][slot_idx] = v(j);

            // (4) set lambda from eig
            lambda[slot_idx] = eigs(i);
          }

          // (3.b) map complex conjugate pair onto A... [i - chain_len, ...]
          auto slots_conj = indices_equal(eigs, std::conj(eigs(i)));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots_conj);
            if (slot_idx < 0) break;

            auto vbar = map_to_A(U, conjugate(chain(k)));

            for (size_t j = 0; j < N; ++j) V[j][slot_idx] = vbar(j);

            // (4) set lambda from eig
            lambda[slot_idx] = std::conj(eigs(i));
          }
        } else {
          // (0) setup jordan chain with padded dim to defect to ensure valid
          // block creation
          const int chain_len = std::min(1 + defect, 1 + i);
          Vector<Vector<Complex, N>, N> chain;

          // (1) seed and back solve for homogenous case (H - \lambda I) x = 0
          seed_real_block(i, chain[0]);
          back_sub_1x1_block(H, i, chain[0]);

          // (2) back solve for each (H - \lambda I) x_k = x_{k+1}
          for (int k = 1; k < chain_len; ++k)
            back_sub_shifted_rhs(H, eigs(i), chain(k - 1), chain[k]);

          // (3) map back to A
          auto slots = indices_equal(eigs, eigs(i));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots);
            if (slot_idx < 0) break;

            auto v = map_to_A(U, chain[k]);

            for (size_t j = 0; j < N; ++j) V[j][slot_idx] = v(j);

            // (4) set lambda from eig
            lambda[slot_idx] = eigs(i);
          }
        }
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
  constexpr void back_sub_1x1_block(const Mat& H, const int& idx,
                                    Vector<Complex, N>& x) {
    // (dakre) solve (H - \lambda I) x = 0 by way of back sub from idx to top
    // (H_rr - \Lambda) x_r + \sum_0^idx H_rc * x_c = 0 ... isolate x_r and
    // solve
    for (int r = idx - 1; r >= 0; --r) {
      Complex sum(0);
      for (int c = r + 1; c <= idx; ++c) sum += Complex(H(r)(c)) * x(c);

      const auto denom = Complex(H(r)(r)) - lambda(idx);
      if (denom == Complex(0))
        x[r] = Complex(0);
      else
        x[r] = -sum / denom;
    }
  }

  /**
   * @brief back sub for shifted repeated \lambda case
   *
   * @param matrix of type T with N x N elements
   * @param complex \lambda
   * @param rhs v_{k-1} eigen vector
   * @param mutable eigen vector (this stages update)
   *
   * @return (void)
   */
  constexpr void back_sub_shifted_rhs(const Mat& H, const Complex& eig,
                                      const Vector<Complex, N>& rhs,
                                      Vector<Complex, N>& x) {
    // (dakre) back solve (H - \lambda I) x = rhs
    for (int r = int(N) - 1; r >= 0; --r) {
      Complex sum(0);
      for (int c = r + 1; c < int(N); ++c) sum += H(r)(c) * x(c);
      const auto denom = Complex(H(r)(r)) - eig;
      if (denom == Complex(0))
        x[r] = Complex(0);
      else
        x[r] = (rhs(r) - sum) / denom;
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
  constexpr void seed_complex_block(const Mat& H, const int& idx,
                                    Vector<Complex, N>& x) {
    const auto& a = H(idx - 1)(idx - 1);
    const auto& b = H(idx - 1)(idx);
    const auto& c = H(idx)(idx - 1);
    const auto& d = H(idx)(idx);

    // seed eigenvector at position idx to kick of back substitution
    const Complex la = lambda[idx] - Complex(a);
    Complex z0, z1;
    if (L2<Vector<T, 1>>()(Vector<T, 1>{b}) >= L2<Vector<Complex, 1>>()(la)) {
      // Use [b; lambda - a]
      z0 = Complex(b);
      z1 = la;
    } else {
      // Use [lambda - d; c]
      z0 = lambda[idx] - Complex(d);
      z1 = Complex(c);
    }
    Real s = constexpr_sqrt(L2<Vector<Complex, 1>>()(z0) +
                            L2<Vector<Complex, 1>>()(z1));
    if (s > Real(0)) {
      z0 /= s;
      z1 /= s;
    }

    x[idx - 1] = z0;
    x[idx] = z1;
  }

  /**
   * @brief seed real eigenvector
   *
   * @param mutable eigenvector x
   *
   * @return (void)
   */
  constexpr void seed_real_block(const int& idx, Vector<Complex, N>& x) {
    for (size_t i = 0; i < N; ++i) x[i] = Complex(0);
    x[idx] = Complex(1);
  }

  /**
   * @brief map eigenvector from H to A
   *
   * @param unitary basis U from real schur upper triangular form on H
   * @param eigenvector v
   *
   * @return eigenvector in basis A
   */
  constexpr auto map_to_A(const Mat& U, const Vector<Complex, N>& x) {
    Vector<Complex, N> v;

    for (size_t j = 0; j < N; ++j) {
      Complex sum(0);
      for (size_t k = 0; k < N; ++k) sum += Complex(U(j)(k)) * x(k);
      v[j] = sum;
    }

    return normalize(v, L2<Vector<Complex, N>>());
  }
};

}  // namespace linalg
