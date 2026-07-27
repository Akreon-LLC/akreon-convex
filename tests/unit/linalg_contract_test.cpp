/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#include <complex>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include "linalg/linalg.hpp"
#include "linalg/norms/norm_traits.hpp"
#include "linalg/operator/operator_traits.hpp"

namespace {

using RealVector = linalg::Vector<double, 3>;
using ComplexVector = linalg::Vector<std::complex<double>, 3>;
using RectangularMatrix = linalg::Matrix<double, 2, 3>;
using SquareMatrix = linalg::Matrix<double, 3, 3>;
using RealSpace = linalg::RowVector<double, 3>;
using ComplexSpace = linalg::RowVector<std::complex<double>, 3>;
using Coordinates3 = linalg::CoordinateSpace<double, 3>;
using MatrixCoordinates = linalg::MatrixSpace<double, 2, 3>;
using SymmetricCoordinates = linalg::SymmetricMatrixSpace<double, 2>;

struct UnsupportedScalar {};

TEST_CASE("linalg compile-time contracts", "[linalg][contracts]") {
  STATIC_REQUIRE(linalg::Scalar<double>);
  STATIC_REQUIRE(linalg::Scalar<std::complex<double>>);
  STATIC_REQUIRE(!linalg::Scalar<int>);
  STATIC_REQUIRE(!linalg::Scalar<UnsupportedScalar>);

  STATIC_REQUIRE(linalg::Field<double>);
  STATIC_REQUIRE(linalg::Field<const double>);
  STATIC_REQUIRE(linalg::Field<volatile double&>);
  STATIC_REQUIRE(linalg::Field<std::complex<double>>);
  STATIC_REQUIRE(linalg::Field<const std::complex<float>&>);
  STATIC_REQUIRE(!linalg::Field<int>);
  STATIC_REQUIRE(!linalg::Field<UnsupportedScalar>);

  STATIC_REQUIRE(linalg::Real<double>);
  STATIC_REQUIRE(linalg::Real<const float&>);
  STATIC_REQUIRE(!linalg::Real<std::complex<double>>);
  STATIC_REQUIRE(linalg::Complex<std::complex<double>>);
  STATIC_REQUIRE(linalg::Complex<const std::complex<float>&>);
  STATIC_REQUIRE(!linalg::Complex<double>);

  STATIC_REQUIRE(std::same_as<linalg::scalar_type_t<const double&>, double>);
  STATIC_REQUIRE(
      std::same_as<linalg::scalar_type_t<const std::complex<double>&>,
                   std::complex<double>>);
  STATIC_REQUIRE(
      std::same_as<linalg::real_type_t<const std::complex<double>&>, double>);
  STATIC_REQUIRE(linalg::is_complex_v<const std::complex<double>&>);
  STATIC_REQUIRE(!linalg::is_complex_v<const double&>);
  STATIC_REQUIRE(linalg::sqrt2_v<double> > 1.4);
  STATIC_REQUIRE(linalg::inv_sqrt2_v<double> > 0.7);
  STATIC_REQUIRE(linalg::ln2_v<double> > 0.0);

  STATIC_REQUIRE(linalg::VectorContainer<RealVector>);
  STATIC_REQUIRE(linalg::VectorContainer<const RealVector&>);
  STATIC_REQUIRE(linalg::VectorContainer<ComplexVector>);
  STATIC_REQUIRE(!linalg::VectorContainer<double>);
  STATIC_REQUIRE(linalg::FieldVector<RealVector>);
  STATIC_REQUIRE(linalg::RealVector<const RealVector&>);
  STATIC_REQUIRE(!linalg::ComplexVector<RealVector>);
  STATIC_REQUIRE(linalg::ComplexVector<const ComplexVector&>);
  STATIC_REQUIRE(!linalg::RealVector<ComplexVector>);
  STATIC_REQUIRE(linalg::vector_dim_v<const RealVector&> == 3);
  STATIC_REQUIRE(
      std::same_as<linalg::vector_element_t<const RealVector&>, double>);
  STATIC_REQUIRE(!linalg::VectorContainer<RectangularMatrix>);
  STATIC_REQUIRE(!linalg::FieldVector<RectangularMatrix>);
  STATIC_REQUIRE(
      std::same_as<linalg::matrix_scalar_t<RectangularMatrix>, double>);

  STATIC_REQUIRE(linalg::MatrixContainer<RectangularMatrix>);
  STATIC_REQUIRE(linalg::MatrixContainer<const RectangularMatrix&>);
  STATIC_REQUIRE(!linalg::SquareMatrix<RectangularMatrix>);
  STATIC_REQUIRE(linalg::SquareMatrix<SquareMatrix>);
  STATIC_REQUIRE(linalg::SquareMatrix<const SquareMatrix&>);
  STATIC_REQUIRE(linalg::RealMatrix<const RectangularMatrix&>);
  STATIC_REQUIRE(!linalg::ComplexMatrix<RectangularMatrix>);
  STATIC_REQUIRE(
      linalg::ComplexMatrix<linalg::Matrix<std::complex<double>, 2, 2>>);
  STATIC_REQUIRE(linalg::mat_rows_v<const RectangularMatrix&> == 2);
  STATIC_REQUIRE(linalg::mat_cols_v<const RectangularMatrix&> == 3);

  using DeducedMatrix = decltype(linalg::Matrix{linalg::Vector{1.0, 2.0},
                                                linalg::Vector{3.0, 4.0}});
  STATIC_REQUIRE(std::same_as<DeducedMatrix, linalg::Matrix<double, 2, 2>>);
  STATIC_REQUIRE(
      std::same_as<decltype(std::declval<RectangularMatrix&>()(0, 0)),
                   double&>);
  STATIC_REQUIRE(
      std::same_as<decltype(std::declval<const RectangularMatrix&>()(0, 0)),
                   const double&>);

  STATIC_REQUIRE(linalg::CompatibleVectors<RealVector, const RealVector&>);
  STATIC_REQUIRE(!linalg::CompatibleVectors<RealVector, ComplexVector>);
  STATIC_REQUIRE(linalg::MatVecCompatible<RectangularMatrix, RealVector>);
  STATIC_REQUIRE(
      !linalg::MatVecCompatible<RectangularMatrix, linalg::Vector<double, 2>>);
  STATIC_REQUIRE(linalg::MatMulCompatible<RectangularMatrix,
                                          linalg::Matrix<double, 3, 4>>);
  STATIC_REQUIRE(!linalg::MatMulCompatible<RectangularMatrix,
                                           linalg::Matrix<double, 2, 4>>);

  STATIC_REQUIRE(linalg::VectorSpace<RealSpace>);
  STATIC_REQUIRE(linalg::VectorSpace<const RealSpace&>);
  STATIC_REQUIRE(linalg::RnSpace<RealSpace>);
  STATIC_REQUIRE(!linalg::RnSpace<ComplexSpace>);
  STATIC_REQUIRE(linalg::space_dim_v<const RealSpace&> == 3);
  STATIC_REQUIRE(std::same_as<linalg::space_type_t<const RealSpace&>, double>);
  STATIC_REQUIRE(
      std::same_as<linalg::space_point_t<const RealSpace&>, RealVector>);

  STATIC_REQUIRE(linalg::CoordinatizedSpace<Coordinates3>);
  STATIC_REQUIRE(linalg::CoordinatizedSpace<MatrixCoordinates>);
  STATIC_REQUIRE(linalg::CoordinatizedSpace<SymmetricCoordinates>);
  STATIC_REQUIRE(linalg::space_dim_v<const Coordinates3&> == 3);
  STATIC_REQUIRE(linalg::space_dim_v<const MatrixCoordinates&> == 6);
  STATIC_REQUIRE(linalg::space_dim_v<const SymmetricCoordinates&> == 3);
  STATIC_REQUIRE(std::same_as<linalg::space_element_t<MatrixCoordinates>,
                              RectangularMatrix>);
  STATIC_REQUIRE(std::same_as<linalg::space_coordinate_t<MatrixCoordinates>,
                              linalg::Vector<double, 6>>);

  using EuclideanNorm = linalg::L2<RealVector>;
  STATIC_REQUIRE(linalg::NormLike<EuclideanNorm>);
  STATIC_REQUIRE(linalg::NormLike<const EuclideanNorm&>);
  STATIC_REQUIRE(
      std::same_as<linalg::norm_type_t<const EuclideanNorm&>, double>);
  STATIC_REQUIRE(linalg::norm_size_v<const EuclideanNorm&> == 3);
  STATIC_REQUIRE(!linalg::NormLike<UnsupportedScalar>);

  STATIC_REQUIRE(linalg::LinearOperatorLike<SquareMatrix>);
  STATIC_REQUIRE(linalg::LinearOperatorLike<const SquareMatrix&>);
}

} // namespace

TEST_CASE("coordinate maps round trip", "[linalg][contracts][space]") {
  constexpr RectangularMatrix matrix{linalg::Vector{1.0, 2.0, 3.0},
                                     linalg::Vector{4.0, 5.0, 6.0}};
  constexpr auto coordinate = linalg::coordinates<MatrixCoordinates>(matrix);
  STATIC_REQUIRE(coordinate == linalg::Vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  STATIC_REQUIRE(linalg::from_coordinates<MatrixCoordinates>(coordinate) ==
                 matrix);

  constexpr linalg::Matrix<double, 2, 2> symmetric{linalg::Vector{2.0, 1.0},
                                                   linalg::Vector{1.0, 2.0}};
  constexpr linalg::space_element_t<SymmetricCoordinates> element{symmetric};
  constexpr auto svec = linalg::coordinates<SymmetricCoordinates>(element);
  STATIC_REQUIRE(svec(0) == 2.0);
  STATIC_REQUIRE(svec(2) == 2.0);
  constexpr auto reconstructed =
      linalg::from_coordinates<SymmetricCoordinates>(svec);
  STATIC_REQUIRE(linalg::coordinates<SymmetricCoordinates>(reconstructed) ==
                 svec);
}
