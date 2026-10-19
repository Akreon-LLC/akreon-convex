/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <akreon/linalg/linalg.hpp>
#include <akreon/convex/convex.hpp>
#include <akreon/convex/test_helpers.hpp>

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;

TEST_CASE("Trace objective", "[convex_fn]") {
  SECTION("Full square matrix trace") {
    using Space = Matrixd<2, 2>;
    constexpr auto c = Trace<Space>::weights();
    constexpr auto c_exp = Vectord<4>{1.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(c == c_exp);
  }

  SECTION("Symmetric matrix trace follows svec diagonal coordinates") {
    using Mat = Matrixd<3, 3>;
    using Space = SymmetricMatrix<Mat>;
    constexpr auto c = Trace<Space>::weights();
    constexpr auto c_exp = Vectord<6>{1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(c == c_exp);
  }

  SECTION("Trace block objective inside merged variable") {
    using MergedSpace = RowVectord<8>;
    using TraceSpace = SymmetricMatrix<Matrixd<2, 2>>;
    constexpr auto X = Variable<VarMeta<MergedSpace>>();
    constexpr auto obj = make_trace_block<2, TraceSpace>(X);
    constexpr auto prob =
        make_problem(X)
            .in_set(Hyperplane(
                Vectord<8>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 0.0, X))
            .minimize(obj);

    constexpr auto c = prob.linear_objective();
    constexpr auto c_exp = Vectord<8>{0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    STATIC_REQUIRE(c == c_exp);
  }
}

TEST_CASE("Abs evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto abs_fn = Abs(x);
  STATIC_REQUIRE(abs_fn(Vectord<1>{0.0}) == 0.0);
  STATIC_REQUIRE(abs_fn(Vectord<1>{-2.0}) == 2.0);

  constexpr mat_traits<Matrix<double, 5, 3>> F_mat;
  constexpr auto F = F_mat(1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0,
                           1.0, 0.0, 0.0, 0.0, 1.0);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = abs_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<5>{}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 5);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto abs_concave = abs_fn.mul<-2.0>().add(1.0);
  STATIC_REQUIRE(abs_concave(Vectord<1>{0.0}) == 1.0);
  STATIC_REQUIRE(abs_concave(Vectord<1>{-2.0}) == -3.0);

  constexpr mat_traits<Matrix<double, 5, 3>> F1_mat;
  constexpr auto F1 = F1_mat(1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, -1.0,
                             0.0, 1.0, 0.0, 0.0, 0.0, -1.0);

  constexpr auto builder1 = abs_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  constexpr linalg::Vector<double, 3> coord2{0.0, 0.0, 2.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord2), canon_map1.linear_part()));
  STATIC_REQUIRE(canon_map1.translation()(0) == 1.0);
  STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<5>{}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 5);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("Exp evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto exp_fn = Exp(x);
  STATIC_REQUIRE(exp_fn(Vectord<1>{0.0}) == 1.0);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = exp_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
  constexpr auto F = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  STATIC_REQUIRE(canon_cb.map ==
                 affine::AffineMap(F, Vectord<3>{0.0, 1.0, 0.0}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 1);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto exp_concave = exp_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(exp_concave(Vectord<1>{0.0}) == -1.0);

  constexpr auto F1 = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
  constexpr auto builder1 = exp_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(canon_cb1.map ==
                 affine::AffineMap(F1, Vectord<3>{0.0, 1.0, 0.0}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 1);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("Geometric mean evaluates and canonicalizes affine compositions",
          "[concave_fn]") {
  using Space = Matrixd<1, 4>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto geo_mean = GeoMean(x);
  STATIC_REQUIRE(geo_mean(Vectord<4>{1.0, 1.0, 1.0, 1.0}) == 1.0);
  STATIC_REQUIRE(
      constexpr_abs(geo_mean(Vectord<4>{81.0, 1.0, 1.0, 1.0}) - 3.0) <= eps);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = geo_mean.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 5> coord{0.0, 0.0, 0.0, 0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  constexpr auto F = make_identity<double, 5>();
  STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<5>{}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 1);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto geo_mean_convex = geo_mean.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(geo_mean_convex(Vectord<4>{1.0, 1.0, 1.0, 1.0}) == -1.0);

  constexpr mat_traits<Matrix<double, 5, 5>> F_mat;
  constexpr auto F1 =
      F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0);

  constexpr auto builder1 = geo_mean_convex.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<5>{}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 1);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("InvPos evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto inv_pos_fn = InvPos(x);
  STATIC_REQUIRE(inv_pos_fn(Vectord<1>{1.0}) == 1.0);
  STATIC_REQUIRE(inv_pos_fn(Vectord<1>{2.0}) == 0.5);

  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;
  constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
  constexpr auto F =
      F_mat(0.0, 0.0, inv_sqrt2, -inv_sqrt2, inv_sqrt2, inv_sqrt2);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = inv_pos_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  STATIC_REQUIRE(
      canon_cb.map ==
      affine::AffineMap(F, Vectord<3>{constexpr_sqrt_safe(2.0), 0.0, 0.0}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 1);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto inv_pos_concave = inv_pos_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(inv_pos_concave(Vectord<1>{1.0}) == -1.0);
  STATIC_REQUIRE(inv_pos_concave(Vectord<1>{2.0}) == -0.5);

  constexpr mat_traits<Matrix<double, 3, 2>> F1_mat;
  constexpr auto F1 =
      F1_mat(0.0, 0.0, inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2);

  constexpr auto builder1 = inv_pos_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(
      canon_cb1.map ==
      affine::AffineMap(F1, Vectord<3>{constexpr_sqrt_safe(2.0), 0.0, 0.0}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 1);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("Log evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<PosVarMeta<Space>>();
  constexpr auto log_fn = Log(x);
  STATIC_REQUIRE(log_fn(Vectord<1>{1.0}) == 0.0);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = log_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
  constexpr auto F = F_mat(0.0, 1.0, 0.0, 0.0, 1.0, 0.0);

  constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  STATIC_REQUIRE(canon_cb.map ==
                 affine::AffineMap(F, Vectord<3>{0.0, 1.0, 0.0}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 1);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto log_convex = log_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(log_convex(Vectord<1>{1.0}) == 0.0);

  constexpr auto F1 = F_mat(0.0, -1.0, 0.0, 0.0, 1.0, 0.0);

  constexpr auto builder1 = log_convex.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(canon_cb1.map ==
                 affine::AffineMap(F1, Vectord<3>{0.0, 1.0, 0.0}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 1);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("Max evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 5>;
  constexpr auto var = Variable<VarMeta<Space>>();
  constexpr auto x = Vectord<5>{-1.0, 4.5, 2.0, 3.0, 5.0};
  constexpr auto max_fn = Max(var);
  STATIC_REQUIRE(max_fn.eval(x) == 5.0);

  constexpr mat_traits<Matrix<double, 5, 6>> F_mat;
  constexpr auto F = F_mat(-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                           0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = max_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 6> coord{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<5>{}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 1);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto max_concave = max_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(max_concave(x) == -5.0);

  constexpr auto F1 =
      F_mat(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
            0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, -1.0, -1.0);

  constexpr auto builder1 = max_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<5>{}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 1);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("Norm evaluates and canonicalizes L1, L2 and Linf functions",
          "[convex_fn]") {
  using Space = Matrixd<1, 2>;
  constexpr auto var = Variable<VarMeta<Space>>();
  using Codom = Matrixd<1, 1>;

  SECTION("Test L2 fn") {
    constexpr auto norm =
        Norm<NormMeta<Space, Codom>, decltype(var), L2<Vectord<2>>>(var);
    STATIC_REQUIRE(norm.eval(Vectord<2>{}) <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}) == constexpr_sqrt_safe(5.0));

    constexpr mat_traits<Matrix<double, 3, 3>> F_mat;
    constexpr auto F = F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = norm.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<3>{}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto norm_concave = norm.mul<-1.0>().add(0.0);
    constexpr auto builder1 = norm_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;

    constexpr auto F1 = F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<3>{}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(canon_cb1.validate());
  }

  SECTION("Test Linf fn") {
    constexpr auto norm =
        Norm<NormMeta<Space, Codom>, decltype(var), Linf<Vectord<2>>>(var);
    STATIC_REQUIRE(norm.eval(Vectord<2>{}) <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}) == 2.0);

    constexpr mat_traits<Matrix<double, 4, 3>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = norm.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<4>{}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto norm_concave = norm.mul<-1.0>().add(0.0);
    constexpr auto builder1 = norm_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;

    constexpr auto F1 =
        F_mat(1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0);
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<4>{}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 2);
    STATIC_REQUIRE(canon_cb1.validate());
  }

  SECTION("Test L1 fn") {
    constexpr auto norm =
        Norm<NormMeta<Space, Codom>, decltype(var), L1<Vectord<2>>>(var);
    STATIC_REQUIRE(norm.eval(Vectord<2>{}) <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}) == 3.0);

    constexpr mat_traits<Matrix<double, 7, 5>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = norm.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 5> coord{0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<7>{}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 4);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto norm_concave = norm.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(norm_concave(Vectord<2>{1.0, 2.0}) == -3.0);

    constexpr auto F1 =
        F_mat(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    constexpr auto builder1 = norm_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<7>{}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 4);
    STATIC_REQUIRE(canon_cb1.validate());
  }
}

TEST_CASE("Square evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto square_fn = Square(x);
  STATIC_REQUIRE(square_fn(Vectord<1>{0.0}) == 0.0);
  STATIC_REQUIRE(square_fn(Vectord<1>{2.0}) == 4.0);

  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;
  constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
  constexpr auto F = F_mat(1.0, 0.0, 0.0, -inv_sqrt2, 0.0, inv_sqrt2);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = square_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  STATIC_REQUIRE(
      canon_cb.map ==
      affine::AffineMap(F, Vectord<3>{0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 1);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto square_concave = square_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(square_concave(Vectord<1>{0.0}) == 0.0);
  STATIC_REQUIRE(square_concave(Vectord<1>{1.0}) == -1.0);

  constexpr auto F1 = F_mat(1.0, 0.0, 0.0, inv_sqrt2, 0.0, -inv_sqrt2);
  constexpr auto builder1 = square_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(
      canon_cb1.map ==
      affine::AffineMap(F1, Vectord<3>{0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 1);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("LogSumExp evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 4>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto lse = LogSumExp(x);
  STATIC_REQUIRE(constexpr_abs(lse(Vectord<4>{}) - 1.38629436112) < eps);
  STATIC_REQUIRE(constexpr_abs(lse(Vectord<4>{-1.0, 0.0, 0.0, 0.0}) -
                               1.21428330036) < eps);

  constexpr mat_traits<Matrix<double, 13, 9>> F_mat;
  constexpr auto F = F_mat(
      1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0,
      -1.0);
  constexpr auto g = linalg::Vector<double, 13>{
      0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = lse.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 9> coord{0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 1.0, 0.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));
  STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, g));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 2);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto lse_concave = lse.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(constexpr_abs(lse_concave(Vectord<4>{}) + 1.38629436112) <
                 eps);

  constexpr auto F1 = F_mat(
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0);

  constexpr auto builder1 = lse_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, g));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 2);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("Sqrt evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<NonnegVarMeta<Space>>();
  constexpr auto sqrt_fn = Sqrt(x);
  STATIC_REQUIRE(sqrt_fn(Vectord<1>{0.0}) == constexpr_sqrt_safe(0.0));
  STATIC_REQUIRE(sqrt_fn(Vectord<1>{2.0}) == constexpr_sqrt_safe(2.0));

  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;
  constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
  constexpr auto F = F_mat(0.0, 1.0, -inv_sqrt2, 0.0, inv_sqrt2, 0.0);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = sqrt_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;
  constexpr auto canon_cb = new_builder.cb;

  constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  STATIC_REQUIRE(
      canon_cb.map ==
      affine::AffineMap(F, Vectord<3>{0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2}));

  constexpr auto cones = canon_cb.cones;
  constexpr auto sz = std::tuple_size<decltype(cones)>::value;
  STATIC_REQUIRE(sz == 1);
  STATIC_REQUIRE(canon_cb.validate());

  constexpr auto sqrt_concave = sqrt_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(sqrt_concave(Vectord<1>{0.0}) ==
                 -1.0 * constexpr_sqrt_safe(0.0));
  STATIC_REQUIRE(sqrt_concave(Vectord<1>{1.0}) ==
                 -1.0 * constexpr_sqrt_safe(1.0));

  constexpr auto F1 = F_mat(0.0, -1.0, -inv_sqrt2, 0.0, inv_sqrt2, 0.0);
  constexpr auto builder1 = sqrt_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = builder1.coord_map;
  constexpr auto canon_cb1 = builder1.cb;

  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  STATIC_REQUIRE(
      canon_cb1.map ==
      affine::AffineMap(F1, Vectord<3>{0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2}));

  constexpr auto cones1 = canon_cb1.cones;
  constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
  STATIC_REQUIRE(sz1 == 1);
  STATIC_REQUIRE(canon_cb1.validate());
}

TEST_CASE("Softplus evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto softplus_fn = Softplus(x);
  STATIC_REQUIRE(softplus_fn(Vectord<1>{0.0}) ==
                 constexpr_ln_safe(1 + constexpr_exp(0.0)));

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = softplus_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;

  constexpr linalg::Vector<double, 4> coord{0.0, 1.0, 0.0, 0.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  constexpr auto canon_cb = new_builder.cb;
  STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 3);

  constexpr mat_traits<Matrix<double, 7, 4>> F_mat;
  constexpr auto F = F_mat(0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0);
  STATIC_REQUIRE(all_equal(F, canon_cb.map.linear_part()));
  STATIC_REQUIRE(canon_cb.map.translation() ==
                 Vectord<7>{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0});

  constexpr auto softplus_concave = softplus_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(softplus_concave(Vectord<1>{0.0}) ==
                 -1.0 * constexpr_ln_safe(1 + constexpr_exp(0.0)));

  constexpr auto new_builder1 =
      softplus_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = new_builder1.coord_map;
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  constexpr auto canon_cb1 = new_builder1.cb;
  STATIC_REQUIRE(std::tuple_size<decltype(canon_cb1.cones)>::value == 3);

  constexpr auto F1 = F_mat(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.0);
  STATIC_REQUIRE(all_equal(F1, canon_cb1.map.linear_part()));
  STATIC_REQUIRE(canon_cb1.map.translation() ==
                 Vectord<7>{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0});
}

TEST_CASE("Harmonic mean evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Dom = Matrixd<1, 2>;
  using Codom = Matrixd<1, 1>;

  constexpr auto x = Variable<PosVarMeta<Dom>>();
  constexpr auto hmean_fn = HarmonicMean(x);
  STATIC_REQUIRE(hmean_fn(Vectord<2>{1.0, 1.0}) == 1.0);

  constexpr auto builder = create_empty_cblk_builder<Codom>();
  constexpr auto new_builder = hmean_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;

  constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  constexpr auto canon_cb = new_builder.cb;
  STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 1);

  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;
  constexpr mat_traits<Matrix<double, 3, 3>> F_mat;
  constexpr auto F = F_mat(0.0, 0.0, inv_sqrt2, inv_sqrt2, -inv_sqrt2, 0.0,
                           inv_sqrt2, inv_sqrt2, -inv_sqrt2);
  STATIC_REQUIRE(all_equal(F, canon_cb.map.linear_part()));
  STATIC_REQUIRE(canon_cb.map.translation() == Vectord<3>{});

  constexpr auto hmean_concave = hmean_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(hmean_concave(Vectord<2>{1.0, 1.0}) == -1.0);

  constexpr auto new_builder1 =
      hmean_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = new_builder1.coord_map;
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  constexpr auto canon_cb1 = new_builder1.cb;
  STATIC_REQUIRE(std::tuple_size<decltype(canon_cb1.cones)>::value == 1);

  constexpr auto F1 = F_mat(0.0, 0.0, -inv_sqrt2, inv_sqrt2, -inv_sqrt2, 0.0,
                            inv_sqrt2, inv_sqrt2, inv_sqrt2);
  STATIC_REQUIRE(all_equal(F1, canon_cb1.map.linear_part()));
  STATIC_REQUIRE(canon_cb.map.translation() == Vectord<3>{});
}

TEST_CASE("KL divergence evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Dom = Matrixd<1, 3>;
  using Codom = Matrixd<1, 1>;

  constexpr auto ones = Vectord<3>{1.0, 1.0, 1.0};
  constexpr auto q = Variable<PosVarMeta<Dom>>();

  constexpr auto kldiv_fn = KLDivergence(ones, q);
  STATIC_REQUIRE(kldiv_fn(ones) == 0.0);

  constexpr auto builder = create_empty_cblk_builder<Codom>();
  constexpr auto new_builder = kldiv_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;

  constexpr linalg::Vector<double, 10> coord{0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0, 0.0, 0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  constexpr auto canon_cb = new_builder.cb;
  STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 2);

  constexpr mat_traits<Matrix<double, 10, 10>> F_mat;
  constexpr auto F = F_mat(
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 1.0);
  STATIC_REQUIRE(all_equal(F, canon_cb.map.linear_part()));
  STATIC_REQUIRE(canon_cb.map.translation() == Vectord<10>{});

  constexpr auto kldiv_concave = kldiv_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(kldiv_concave(ones) == 0.0);

  constexpr auto new_builder1 =
      kldiv_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = new_builder1.coord_map;
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  constexpr auto canon_cb1 = new_builder1.cb;
  STATIC_REQUIRE(std::tuple_size<decltype(canon_cb1.cones)>::value == 2);
  constexpr auto F1 = F_mat(
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 1.0);

  STATIC_REQUIRE(all_equal(F1, canon_cb1.map.linear_part()));
  STATIC_REQUIRE(canon_cb1.map.translation() == Vectord<10>{});
}

TEST_CASE("Huber evaluates and canonicalizes affine compositions",
          "[convex_fn]") {
  using Space = Matrixd<1, 1>;
  constexpr auto x = Variable<VarMeta<Space>>();
  constexpr auto delta = 2.0;
  constexpr auto huber_fn = Huber(delta, x);
  STATIC_REQUIRE(huber_fn(Vectord<1>{0.0}) == 0.0);
  STATIC_REQUIRE(huber_fn(Vectord<1>{1.0}) == 0.5);
  STATIC_REQUIRE(huber_fn(Vectord<1>{3.0}) == 4.0);

  constexpr auto builder = create_empty_cblk_builder<Space>();
  constexpr auto new_builder = huber_fn.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map = new_builder.coord_map;

  constexpr linalg::Vector<double, 6> coord{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

  constexpr auto canon_cb = new_builder.cb;
  STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 2);

  constexpr auto inv_sqrt2 = inv_sqrt2_v<double>;
  constexpr mat_traits<Matrix<double, 9, 6>> F_mat;
  constexpr auto F = F_mat(
      1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, -inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, inv_sqrt2, 0.0);
  STATIC_REQUIRE(all_equal(F, canon_cb.map.linear_part()));
  STATIC_REQUIRE(canon_cb.map.translation() ==
                 Vectord<9>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * inv_sqrt2,
                            0.5 * inv_sqrt2});

  constexpr auto huber_concave = huber_fn.mul<-1.0>().add(0.0);
  STATIC_REQUIRE(huber_concave(Vectord<1>{0.0}) == 0.0);
  STATIC_REQUIRE(huber_concave(Vectord<1>{1.0}) == -0.5);
  STATIC_REQUIRE(huber_concave(Vectord<1>{3.0}) == -4.0);

  constexpr auto new_builder1 =
      huber_concave.canonicalize<AtomType::FN>(builder);
  constexpr auto canon_map1 = new_builder1.coord_map;
  STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

  constexpr auto canon_cb1 = new_builder1.cb;
  constexpr auto F1 = F_mat(
      1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, -inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, inv_sqrt2, 0.0);
  STATIC_REQUIRE(all_equal(F1, canon_cb1.map.linear_part()));
  STATIC_REQUIRE(canon_cb1.map.translation() ==
                 Vectord<9>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * inv_sqrt2,
                            0.5 * inv_sqrt2});
}
