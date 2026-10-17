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

#include <algorithm>
#include <cmath>
#include <limits>

#include "linalg/operator/hessenberg.hpp"
#include "qr.hpp"

namespace linalg {

/**
 * @brief EigenDecomposition class definition
 *
 * Computes A V = V diag(lambda) using bounded shifted QR on Hessenberg form.
 * Real matrices may retain isolated conjugate-pair blocks in real Schur form.
 * Hermitian eigenvectors use the accumulated orthonormal similarity basis.
 * Nonconvergence or invalid configuration/input leaves converged false.
 * Coupled Hermitian spectra are ordered by decreasing eigenvalue; diagonal
 * inputs retain their coordinate order. Other spectra have no ordering
 * contract.
 */
template <SquareMatrix Mat>
struct EigenDecomposition {
  // @brief alias for underlying type
  using T = matrix_scalar_t<Mat>;

  // @brief aliasing complex and real types for eigen decomposition
  using Real = real_type_t<T>;
  using Complex = std::conditional_t<linalg::Complex<T>, T, std::complex<Real>>;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static_assert(N > 0, "Eigen decomposition requires a nonempty matrix");

  /// Bounded shifted-QR budget; tol is relative to the largest input entry.
  /// These defaults are specific to eigen solves and do not alter QR/LDL
  /// defaults.
  static constexpr config::FactorConfig<Real> default_config() {
    return {.steps = 256,
            .tol = Real(8) * std::numeric_limits<Real>::epsilon(),
            .floor_eps = std::numeric_limits<Real>::min()};
  }

  /**
   * @brief CTOR
   *
   * @param matrix A of type T with Rows x Cols elements
   */
  constexpr explicit EigenDecomposition(const Mat& A) { solve_qr(A); }

  /**
   * @brief CTOR
   *
   * @param A mat to be spectrally decomposed
   * @param config_in Iteration budget, relative deflation tolerance, and
   * eigenvector normalization floor (FactorConfig).
   */
  constexpr explicit EigenDecomposition(
      const Mat& A, const config::FactorConfig<Real>& config_in)
      : config(config_in) {
    solve_qr(A);
  }

  /**
   * @brief get spectral radius
   *
   * AKA maximum magnitude of a matrix as dictated by its max abs eigenvalue
   *
   * @return rho
   */
  constexpr auto spectral_radius() const {
    Complex rho(0.0);
    for (std::size_t i = 0; i < N; ++i) {
      const auto mag_rho = constexpr_sqrt_safe(rho.real() * rho.real() +
                                               rho.imag() * rho.imag());
      const auto mag_lambda =
          constexpr_sqrt_safe(lambda(i).real() * lambda(i).real() +
                              lambda(i).imag() * lambda(i).imag());
      rho = mag_lambda > mag_rho ? lambda(i) : rho;
    }
    return rho;
  }

  /** @brief eigendecomposition member variables */
  Vector<Complex, N> lambda; ///< eigenvalues
  Matrix<Complex, N, N> V;   ///< eigenvectors
  bool converged{false};     ///< eigenvalue convergence flag

private:
  // Shift radicands may be tiny even after scaling a matrix with separated
  // spectral blocks. Use relative squared-residual accuracy for these roots.
  static constexpr Real spectral_sqrt(Real value) {
    return constexpr_sqrt_safe(value, Real(0), Real(1),
                               Real(4) * std::numeric_limits<Real>::epsilon() *
                                   value);
  }

  /**
   * @brief solve A X = Lambda X via QR
   *
   * Following the practical QR Algorithm 7.5 in Matrix Computations
   *
   * @param matrix A of type T with N x N elements
   * @param tolerance dictating algorithm convergence
   *
   * @return (void)
   */
  constexpr void solve_qr(const Mat& A) {
    if (!std::isfinite(config.tol) || config.tol <= Real(0) ||
        !std::isfinite(config.floor_eps) || config.floor_eps <= Real(0))
      return;
    for (std::size_t r = 0; r < N; ++r)
      for (std::size_t c = 0; c < N; ++c)
        if (!std::isfinite(constexpr_abs(A(r, c)))) return;

    Real scale = 0;
    for (std::size_t r = 0; r < N; ++r)
      for (std::size_t c = 0; c < N; ++c)
        scale = std::max(scale, constexpr_abs(A(r, c)));
    // Work at unit scale so quadratic shifts and scalar square roots do not
    // inherit an absolute accuracy limit from the magnitude of the input.
    auto normalized = A;
    if (scale > Real(0))
      for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c) normalized(r, c) /= scale;
    // Scalar and 2x2 matrices are already Hessenberg; avoid an empty reduction.
    auto [H, U] = [&] {
      if constexpr (N > 2) {
        const auto reduction = Hessenberg(normalized);
        return std::tuple{reduction.H(), reduction.U()};
      } else return std::tuple{normalized, make_identity<T, N>()};
    }();
    const Real threshold = config.tol;
    const bool hermitian = [&] {
      if constexpr (linalg::Real<T>)
        return all_equal(normalized, transpose(normalized), threshold);
      else
        return all_equal(normalized, conjugate_transpose(normalized),
                         threshold);
    }();
    Vector<Complex, N> eigs;

    // Deflate actual subdiagonals. Hessenberg entries below them are already
    // zero and cannot establish convergence. Real Schur form may retain only
    // isolated 2x2 blocks with complex conjugate eigenvalues.
    auto deflate = [&] {
      for (std::size_t r = 1; r < N; ++r)
        if (constexpr_abs(H(r, r - 1)) <= threshold) H(r, r - 1) = T(0);
      bool done = true;
      for (std::size_t r = 1; r < N; ++r) {
        if (H(r, r - 1) == T(0)) continue;
        if constexpr (linalg::Complex<T>) done = false;
        else {
          const Real half_difference = (H(r - 1, r - 1) - H(r, r)) / 2;
          const Real discriminant =
              half_difference * half_difference + H(r - 1, r) * H(r, r - 1);
          if (hermitian || discriminant >= 0 ||
              (r > 1 && H(r - 1, r - 2) != T(0)) ||
              (r + 1 < N && H(r + 1, r) != T(0)))
            done = false;
        }
      }
      return done;
    };
    converged = deflate();
    for (std::size_t iteration = 0; !converged && iteration < config.steps;
         ++iteration) {
      // Shift at the bottom of the remaining active block. For real roots use
      // the trailing 2x2 root nearest its last diagonal (Wilkinson shift).
      std::size_t end = N;
      while (end > 1 && H(end - 1, end - 2) == T(0)) --end;
      if constexpr (linalg::Real<T>) {
        while (!hermitian && end > 2 && H(end - 2, end - 3) == T(0)) {
          const Real d = (H(end - 2, end - 2) - H(end - 1, end - 1)) / 2;
          if (d * d + H(end - 2, end - 1) * H(end - 1, end - 2) >= 0) break;
          end -= 2;
          while (end > 1 && H(end - 1, end - 2) == T(0)) --end;
        }
      }
      T shift = H(end - 1, end - 1);
      if constexpr (linalg::Real<T>) {
        if (end > 1) {
          const Real a = H(end - 2, end - 2), d = H(end - 1, end - 1);
          const Real delta = (a - d) / 2;
          const Real discriminant =
              delta * delta + H(end - 2, end - 1) * H(end - 1, end - 2);
          if (discriminant >= 0) {
            const Real root = spectral_sqrt(discriminant);
            shift = (a + d) / 2 + (delta >= 0 ? -root : root);
          }
        }
      }
      if constexpr (linalg::Complex<T>) {
        if (end > 1) {
          const Complex center =
              (H(end - 2, end - 2) + H(end - 1, end - 1)) / Real(2);
          const Complex delta =
              (H(end - 2, end - 2) - H(end - 1, end - 1)) / Real(2);
          const Complex disc =
              delta * delta + H(end - 2, end - 1) * H(end - 1, end - 2);
          const Real radius = constexpr_abs(disc);
          const Complex root(
              spectral_sqrt(
                  std::max(Real(0), (radius + disc.real()) / Real(2))),
              (disc.imag() < Real(0) ? Real(-1) : Real(1)) *
                  spectral_sqrt(
                      std::max(Real(0), (radius - disc.real()) / Real(2))));
          const Complex plus = center + root, minus = center - root;
          shift = constexpr_abs(plus - shift) < constexpr_abs(minus - shift)
                      ? plus
                      : minus;
        }
      }
      auto shifted = H;
      for (std::size_t j = 0; j < N; ++j) shifted(j, j) -= shift;
      bool double_shift = false;
      T trace = T(0), determinant = T(0);
      if constexpr (linalg::Real<T>) {
        if (end > 1 && !hermitian) {
          const T a = H(end - 2, end - 2), d = H(end - 1, end - 1);
          const T b = H(end - 2, end - 1), c = H(end - 1, end - 2);
          double_shift = (a - d) * (a - d) / 4 + b * c < 0;
          trace = a + d;
          determinant = a * d - b * c;
        }
      }
      if (double_shift) {
        // A real quadratic shift follows a conjugate pair without stagnating
        // on its real part. Apply its QR basis as a similarity transform.
        shifted = H * H - trace * H;
        for (std::size_t j = 0; j < N; ++j) shifted(j, j) += determinant;
      }
      const auto qr = QR(shifted);
      const auto Q = qr.Q();
      if (double_shift) H = transpose(Q) * (H * Q);
      else {
        H = qr.R() * Q;
        for (std::size_t j = 0; j < N; ++j) H(j, j) += shift;
      }
      // QR similarity preserves Hessenberg structure in exact arithmetic.
      // Do not let roundoff below the band seed new active blocks later.
      for (std::size_t r = 2; r < N; ++r)
        for (std::size_t c = 0; c + 1 < r; ++c) H(r, c) = T(0);
      U = U * Q;
      converged = deflate();
    }
    if (!converged) return;

    for (std::size_t j = 0; j < N; ++j) {
      eigs(j) = Complex(H(j, j));
      if constexpr (linalg::Real<T>) {
        if (j + 1 < N && H(j + 1, j) != T(0)) {
          const Real center = (H(j, j) + H(j + 1, j + 1)) / 2;
          const Real delta = (H(j, j) - H(j + 1, j + 1)) / 2;
          const Real discriminant = delta * delta + H(j, j + 1) * H(j + 1, j);
          const Real imaginary = spectral_sqrt(-discriminant);
          eigs(j) = Complex(center, imaginary);
          eigs(j + 1) = Complex(center, -imaginary);
          ++j;
        }
      }
    }
    // Back substitution and complex-block seeding read lambda; populate every
    // entry before either operation rather than after constructing its vector.
    lambda = eigs;
    if (hermitian) {
      // The accumulated similarity basis is already orthonormal. Back solving
      // nearly repeated eigenvalues would unnecessarily destroy that property.
      V = as_complex_matrix(U);
      // Preserve diagonal input order; order a coupled Hermitian spectrum by
      // decreasing eigenvalue, as in the existing two-dimensional interface.
      bool diagonal_input = true;
      for (std::size_t r = 0; r < N; ++r)
        for (std::size_t c = 0; c < N; ++c)
          if (r != c && A(r, c) != T(0)) diagonal_input = false;
      if (!diagonal_input) {
        for (std::size_t j = 0; j < N; ++j) {
          std::size_t largest = j;
          for (std::size_t k = j + 1; k < N; ++k)
            if (lambda(k).real() > lambda(largest).real()) largest = k;
          std::swap(lambda(j), lambda(largest));
          for (std::size_t r = 0; r < N; ++r) std::swap(V(r, j), V(r, largest));
        }
      }
      for (std::size_t j = 0; j < N; ++j) lambda(j) *= scale;
      return;
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
    //  - solve for root (H - lambda I) x = 0
    //  - apply coordinate map to x (evec) from unitary basis U to get it
    //  aligned in A
    // b. solve jordan block (AM > 1)
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
      if (used[i]) continue;

      // Jordan block
      // - AM = algebraic multiplicity (# of repeated lambda)
      // - GM = geometric multiplicity (# of linearly independent
      // eigenvectors for lambda)
      // - Defect = # of extra generalized evecs to span the space the Jordan
      // block lies in
      int AM = 0;
      std::for_each(eigs.begin(), eigs.end(), [&](const auto& val) {
        AM += constexpr_abs(eigs(i) - val) < config.tol;
      });
      int GM = 0, defect = 0;
      if (AM > 1) {
        const auto I = make_identity<Complex, N>();
        GM = N - QR(as_complex_matrix(normalized) - eigs(i) * I).rank();
        defect = std::max(0, AM - GM);
      }

      if (0 == defect) {
        Vector<Complex, N> x;

        // (a) solve normally
        if (linalg::Real<T> && eigs(i).imag() != Real(0) && i > 0) {
          // (0) seed & back solve homogenous case (H - lambda I) x = 0
          seed_complex_block(H, i, x);
          back_sub_1x1_block(H, i, x);

          // (1) map vec(s)... conjugate pairs onto A
          auto v = map_to_A(U, x);
          for (std::size_t j = 0; j < N; ++j) V(j, i) = v(j);

          i--;

          v = map_to_A(U, conjugate(x));
          for (std::size_t j = 0; j < N; ++j) V(j, i) = v(j);

          // (2) set eig to lambda
          lambda(i) = eigs(i);

          used[i] = true;
        } else {
          // (0) seed & back solve homogenous case (H - lambda I) x = 0
          seed_real_block(i, x);
          back_sub_1x1_block(H, i, x);

          // (1) map vec onto A
          auto v = map_to_A(U, x);
          for (std::size_t j = 0; j < N; ++j) V(j, i) = v(j);

          // (2) set eig to lambda
          lambda(i) = eigs(i);

          used[i] = true;
        }
      } else {
        // (b) solve for jordan block
        if (linalg::Real<T> && eigs(i).imag() != Real(0) && i > 0) {
          // (0) setup jordan chain with padded dim to defect to ensure valid
          // block creation
          const int chain_len = std::min(1 + defect, (i + 1) / 2);
          Vector<Vector<Complex, N>, N> chain;

          // (1) seed and back solve for homogenous case (H - lambda I) x = 0
          seed_complex_block(H, i, chain(0));
          back_sub_1x1_block(H, i, chain(0));

          // (2) back solve for each (H - lambda I) x_k = x_{k+1}
          for (int k = 1; k < chain_len; ++k)
            back_sub_shifted_rhs(H, lambda(i), chain(k - 1), chain(k));

          // (3) map back to A... [i, i-1, ...]
          auto slots = indices_equal(eigs, eigs(i));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots);
            if (slot_idx < 0) break;

            auto v = map_to_A(U, chain(k));

            for (std::size_t j = 0; j < N; ++j) V(j, slot_idx) = v(j);

            // (4) set lambda from eig
            lambda(slot_idx) = eigs(i);
          }

          // (3.b) map complex conjugate pair onto A... [i - chain_len, ...]
          auto slots_conj = indices_equal(eigs, std::conj(eigs(i)));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots_conj);
            if (slot_idx < 0) break;

            auto vbar = map_to_A(U, conjugate(chain(k)));

            for (std::size_t j = 0; j < N; ++j) V(j, slot_idx) = vbar(j);

            // (4) set lambda from eig
            lambda(slot_idx) = std::conj(eigs(i));
          }
        } else {
          // (0) setup jordan chain with padded dim to defect to ensure valid
          // block creation
          const int chain_len = std::min(1 + defect, 1 + i);
          Vector<Vector<Complex, N>, N> chain;

          // (1) seed and back solve for homogenous case (H - lambda I) x = 0
          seed_real_block(i, chain(0));
          back_sub_1x1_block(H, i, chain(0));

          // (2) back solve for each (H - lambda I) x_k = x_{k+1}
          for (int k = 1; k < chain_len; ++k)
            back_sub_shifted_rhs(H, eigs(i), chain(k - 1), chain(k));

          // (3) map back to A
          auto slots = indices_equal(eigs, eigs(i));

          for (int k = 0; k < chain_len; ++k) {
            const auto slot_idx = take_last_unused(slots);
            if (slot_idx < 0) break;

            auto v = map_to_A(U, chain(k));

            for (std::size_t j = 0; j < N; ++j) V(j, slot_idx) = v(j);

            // (4) set lambda from eig
            lambda(slot_idx) = eigs(i);
          }
        }
      }
    }
    for (std::size_t j = 0; j < N; ++j) lambda(j) *= scale;
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
  constexpr void back_sub_1x1_block(const Mat& H, int idx,
                                    Vector<Complex, N>& x) {
    // (dakre) solve (H - lambda I) x = 0 by way of back sub from idx to top
    // (H_rr - Lambda) x_r + sum_0^idx H_rc * x_c = 0 ... isolate x_r and
    // solve
    for (int r = ((linalg::Real<T> && lambda(idx).imag() != Real(0)) ? idx - 2
                                                                     : idx - 1);
         r >= 0; --r) {
      Complex rhs(0);
      for (int c = r + 1; c <= idx; ++c) rhs -= Complex(H(r, c)) * x(c);
      if constexpr (linalg::Real<T>) {
        if (r > 0 && H(r, r - 1) != T(0)) {
          // Back substitution above an isolated conjugate pair must solve its
          // coupled 2x2 block, rather than treating it as two triangular rows.
          Complex upper_rhs(0);
          for (int c = r + 1; c <= idx; ++c)
            upper_rhs -= Complex(H(r - 1, c)) * x(c);
          const Complex a = Complex(H(r - 1, r - 1)) - lambda(idx);
          const Complex b = Complex(H(r - 1, r));
          const Complex c = Complex(H(r, r - 1));
          const Complex d = Complex(H(r, r)) - lambda(idx);
          const Complex det = a * d - b * c;
          if (det != Complex(0)) {
            x(r - 1) = (d * upper_rhs - b * rhs) / det;
            x(r) = (a * rhs - c * upper_rhs) / det;
          }
          --r;
          continue;
        }
      }
      const auto denom = Complex(H(r, r)) - lambda(idx);
      x(r) = denom == Complex(0) ? Complex(0) : rhs / denom;
    }
  }

  /**
   * @brief back sub for shifted repeated lambda case
   *
   * @param matrix of type T with N x N elements
   * @param complex lambda
   * @param rhs v_{k-1} eigen vector
   * @param mutable eigen vector (this stages update)
   *
   * @return (void)
   */
  constexpr void back_sub_shifted_rhs(const Mat& H, const Complex& eig,
                                      const Vector<Complex, N>& rhs,
                                      Vector<Complex, N>& x) {
    // (dakre) back solve (H - lambda I) x = rhs
    for (int r = int(N) - 1; r >= 0; --r) {
      Complex sum(0);
      for (int c = r + 1; c < int(N); ++c) sum += H(r, c) * x(c);
      const auto denom = Complex(H(r, r)) - eig;
      if (denom == Complex(0)) x(r) = Complex(0);
      else x(r) = (rhs(r) - sum) / denom;
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
  constexpr void seed_complex_block(const Mat& H, int idx,
                                    Vector<Complex, N>& x) {
    const auto& a = H(idx - 1, idx - 1);
    const auto& b = H(idx - 1, idx);
    const auto& c = H(idx, idx - 1);
    const auto& d = H(idx, idx);

    // seed eigenvector at position idx to kick of back substitution
    const Complex la = lambda(idx) - Complex(a);
    Complex z0, z1;
    if (L2<Vector<T, 1>>()(Vector<T, 1>{b}) >= L2<Vector<Complex, 1>>()(la)) {
      // Use [b; lambda - a]
      z0 = Complex(b);
      z1 = la;
    } else {
      // Use [lambda - d; c]
      z0 = lambda(idx) - Complex(d);
      z1 = Complex(c);
    }
    Real s = constexpr_sqrt_safe(L2<Vector<Complex, 1>>()(z0) +
                                 L2<Vector<Complex, 1>>()(z1));
    if (s > Real(0)) {
      z0 /= s;
      z1 /= s;
    }

    x(idx - 1) = z0;
    x(idx) = z1;
  }

  /**
   * @brief seed real eigenvector
   *
   * @param mutable eigenvector x
   *
   * @return (void)
   */
  constexpr void seed_real_block(int idx, Vector<Complex, N>& x) {
    for (std::size_t i = 0; i < N; ++i) x(i) = Complex(0);
    x(idx) = Complex(1);
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

    for (std::size_t j = 0; j < N; ++j) {
      Complex sum(0);
      for (std::size_t k = 0; k < N; ++k) sum += Complex(U(j, k)) * x(k);
      v(j) = sum;
    }

    const auto norm = std::max(L2<Vector<Complex, N>>()(v), config.floor_eps);
    return T(1) / norm * v;
  }

  /** @brief member vars */
  config::FactorConfig<Real> config = default_config();
};

} // namespace linalg
