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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;

TEST_CASE("Test abs function", "[convex_fn]") {
  SECTION("Test abs function") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto abs_fn = Abs<AbsMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(abs_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(abs_fn(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(abs_fn(Vectord<1>{-2.0}).has_value());
    STATIC_REQUIRE(abs_fn(Vectord<1>{-2.0}).value() == 2.0);

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
    STATIC_REQUIRE(abs_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(abs_concave(Vectord<1>{0.0}).value() == 1.0);
    STATIC_REQUIRE(abs_concave(Vectord<1>{-2.0}).has_value());
    STATIC_REQUIRE(abs_concave(Vectord<1>{-2.0}).value() == -3.0);

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
}

TEST_CASE("Test exp fn", "[convex_fn]") {
  SECTION("Test exp fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto exp_fn = Exp<ExpMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(exp_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(exp_fn(Vectord<1>{0.0}).value() == 1.0);

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
    STATIC_REQUIRE(exp_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(exp_concave(Vectord<1>{0.0}).value() == -1.0);

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
}

TEST_CASE("Test geometric mean fn", "[concave_fn]") {
  SECTION("Test geometric mean fn") {
    using Space = Matrixd<1, 4>;
    constexpr auto x = Variable<PosVarMeta<Space>>();
    using Codom = Matrixd<1, 1>;
    constexpr auto geo_mean =
        GeoMean<GeoMeanMeta<Space, Codom>, decltype(x)>(x);
    STATIC_REQUIRE(geo_mean(Vectord<4>{1.0, 1.0, 1.0, 1.0}).has_value());
    STATIC_REQUIRE(geo_mean(Vectord<4>{1.0, 1.0, 1.0, 1.0}).value() == 1.0);
    STATIC_REQUIRE(
        constexpr_abs(geo_mean(Vectord<4>{81.0, 1.0, 1.0, 1.0}).value() -
                      3.0) <= eps);

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
    STATIC_REQUIRE(geo_mean_convex(Vectord<4>{1.0, 1.0, 1.0, 1.0}).has_value());
    STATIC_REQUIRE(geo_mean_convex(Vectord<4>{1.0, 1.0, 1.0, 1.0}).value() ==
                   -1.0);

    constexpr mat_traits<Matrix<double, 5, 5>> F_mat;
    constexpr auto F1 =
        F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0);

    constexpr auto builder1 =
        geo_mean_convex.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;

    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<5>{}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(canon_cb1.validate());
  }
}

TEST_CASE("Test inv pos function", "[convex_fn]") {
  SECTION("Test inv pos fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<PosVarMeta<Space>>();
    constexpr auto inv_pos_fn = InvPos<InvPosMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(!inv_pos_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(!inv_pos_fn(Vectord<1>{-1.0}).has_value());
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{1.0}).value() == 1.0);
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{2.0}).value() == 0.5);

    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = inv_pos_fn.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(
        canon_cb.map ==
        affine::AffineMap(F, Vectord<3>{constexpr_sqrt(2.0), 0.0, 0.0}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto inv_pos_concave = inv_pos_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(!inv_pos_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(!inv_pos_concave(Vectord<1>{-1.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{1.0}).value() == -1.0);
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{2.0}).value() == -0.5);

    constexpr mat_traits<Matrix<double, 3, 2>> F1_mat;
    constexpr auto F1 = F1_mat(0.0, 0.0, 1.0, 0.0, 0.0, -1.0);

    constexpr auto builder1 =
        inv_pos_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;

    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(
        canon_cb1.map ==
        affine::AffineMap(F1, Vectord<3>{constexpr_sqrt(2.0), 0.0, 0.0}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(canon_cb1.validate());
  }
}

TEST_CASE("Test log function", "[convex_fn]") {
  SECTION("Test log fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<PosVarMeta<Space>>();
    constexpr auto log_fn = Log<LogMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(!log_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(log_fn(Vectord<1>{1.0}).value() == 0.0);

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
    STATIC_REQUIRE(!log_convex(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(log_convex(Vectord<1>{1.0}).value() == 0.0);

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
}

TEST_CASE("Test max fn") {
  SECTION("Test max fn") {
    using Space = Matrixd<1, 5>;
    constexpr auto var = Variable<FreeVarMeta<Space>>();
    using Codom = Matrixd<1, 1>;
    constexpr auto x = Vectord<5>{-1.0, 4.5, 2.0, 3.0, 5.0};
    constexpr auto max_fn = Max<MaxMeta<Space, Codom>, decltype(var)>(var);
    STATIC_REQUIRE(max_fn.eval(x).has_value());
    STATIC_REQUIRE(max_fn.eval(x).value() == 5.0);

    constexpr mat_traits<Matrix<double, 5, 6>> F_mat;
    constexpr auto F =
        F_mat(-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, -1.0, 1.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = max_fn.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 6> coord{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<5>{}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 5);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto max_concave = max_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(max_concave(x).value() == -5.0);

    constexpr auto F1 =
        F_mat(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0,
              0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0,
              0.0, 0.0, 0.0, 0.0, -1.0, -1.0);

    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    constexpr auto builder1 = max_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;

    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<5>{}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 5);
    STATIC_REQUIRE(canon_cb1.validate());
  }
}

TEST_CASE("Test norm fn", "[convex_fn]") {
  // constexpr auto map_id = affine::make_map_id<linalg::Matrixd<2, 2>>();
  // constexpr auto cb = create_empty_cblk<double>();

  using Space = Matrixd<1, 2>;
  // constexpr auto cmap_builder = ConeMapBuilder<Space>();

  constexpr auto var = Variable<FreeVarMeta<Space>>();
  using Codom = Matrixd<1, 1>;

  SECTION("Test L2 fn") {
    constexpr auto norm =
        Norm<NormMeta<Space, Codom>, decltype(var), L2<Vectord<2>>>(var);
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() ==
                   constexpr_sqrt(5.0));

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
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 2.0);

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
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 3.0);

    constexpr mat_traits<Matrix<double, 5, 5>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = norm.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 5> coord{0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<5>{}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 3);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto norm_concave = norm.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(norm_concave(Vectord<2>{1.0, 2.0}).value() == -3.0);

    constexpr auto F1 =
        F_mat(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0);

    constexpr auto builder1 = norm_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map == affine::AffineMap(F1, Vectord<5>{}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 3);
    STATIC_REQUIRE(canon_cb1.validate());
  }
}

TEST_CASE("Test square function", "[convex_fn]") {
  SECTION("Test square function") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto square_fn = Square<SquareMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(square_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(square_fn(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(square_fn(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(square_fn(Vectord<1>{2.0}).value() == 4.0);

    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = square_fn.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map ==
                   affine::AffineMap(F, Vectord<3>{0.0, 0.5, 0.0}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto square_concave = square_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(square_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(square_concave(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(square_concave(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(square_concave(Vectord<1>{1.0}).value() == -1.0);

    constexpr auto F1 = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    constexpr auto builder1 =
        square_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;

    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map ==
                   affine::AffineMap(F1, Vectord<3>{0.0, 0.5, 0.0}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(canon_cb1.validate());
  }
}

TEST_CASE("Test log sum exp", "[convex_fn]") {
  SECTION("Test log sum exp") {
    using Space = Matrixd<1, 4>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    using Codom = Matrixd<1, 1>;
    constexpr auto lse = LogSumExp<LogSumExpMeta<Space, Codom>, decltype(x)>(x);
    STATIC_REQUIRE(lse(Vectord<4>{}).has_value());
    STATIC_REQUIRE(constexpr_abs(lse(Vectord<4>{}).value() - 1.38629436112) <
                   eps);
    STATIC_REQUIRE(lse(Vectord<4>{-1.0, 0.0, 0.0, 0.0}).has_value());
    STATIC_REQUIRE(constexpr_abs(lse(Vectord<4>{-1.0, 0.0, 0.0, 0.0}).value() -
                                 1.21428330036) < eps);

    constexpr mat_traits<Matrix<double, 10, 9>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0);
    constexpr auto g = linalg::Vector<double, 10>{0.0, 0.0, 0.0, 0.0, 1.0,
                                                  0.0, 0.0, 0.0, 0.0, 1.0};

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
    STATIC_REQUIRE(lse_concave(Vectord<4>{}).has_value());
    STATIC_REQUIRE(
        constexpr_abs(lse_concave(Vectord<4>{}).value() + 1.38629436112) < eps);

    constexpr auto F1 =
        F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
}

TEST_CASE("Square Root Function", "[convex_fn]") {
  SECTION("Test sqrt function") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<PosVarMeta<Space>>();
    constexpr auto sqrt_fn = Sqrt<SqrtMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(sqrt_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(sqrt_fn(Vectord<1>{0.0}).value() == constexpr_sqrt(0.0));
    STATIC_REQUIRE(sqrt_fn(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(sqrt_fn(Vectord<1>{2.0}).value() == constexpr_sqrt(2.0));

    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(0.0, 1.0, 0.0, 0.0, 1.0, 0.0);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder = sqrt_fn.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map ==
                   affine::AffineMap(F, Vectord<3>{0.0, 0.5, 0.0}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto sqrt_concave = sqrt_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(sqrt_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(sqrt_concave(Vectord<1>{0.0}).value() ==
                   -1.0 * constexpr_sqrt(0.0));
    STATIC_REQUIRE(sqrt_concave(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(sqrt_concave(Vectord<1>{1.0}).value() ==
                   -1.0 * constexpr_sqrt(1.0));

    constexpr auto F1 = F_mat(0.0, -1.0, 0.0, 0.0, 1.0, 0.0);
    constexpr auto builder1 = sqrt_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = builder1.coord_map;
    constexpr auto canon_cb1 = builder1.cb;

    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    STATIC_REQUIRE(canon_cb1.map ==
                   affine::AffineMap(F1, Vectord<3>{0.0, 0.5, 0.0}));

    constexpr auto cones1 = canon_cb1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(canon_cb1.validate());
  }
}

TEST_CASE("Softplus Function", "[convex_fn]") {
  SECTION("Test softplus") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto softplus_fn = Softplus<SoftplusMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(softplus_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(softplus_fn(Vectord<1>{0.0}).value() ==
                   constexpr_ln(1 + constexpr_exp(0.0)).value());

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder =
        softplus_fn.canonicalize<AtomType::FN>(builder);
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
    STATIC_REQUIRE(softplus_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(softplus_concave(Vectord<1>{0.0}).value() ==
                   -1.0 * constexpr_ln(1 + constexpr_exp(0.0)).value());

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
}

TEST_CASE("Harmonic Mean Function", "[convex_fn]") {
  SECTION("Test harmonic mean") {
    using Dom = Matrixd<1, 2>;
    using Codom = Matrixd<1, 1>;

    constexpr auto x = Variable<PosVarMeta<Dom>>();
    constexpr auto hmean_fn =
        HarmonicMean<HarmonicMeanMeta<Dom, Codom>, decltype(x)>(x);
    STATIC_REQUIRE(!hmean_fn(Vectord<2>{}).has_value());
    STATIC_REQUIRE(hmean_fn(Vectord<2>{1.0, 1.0}).has_value());
    STATIC_REQUIRE(hmean_fn(Vectord<2>{1.0, 1.0}).value() == 1.0);

    constexpr auto builder = create_empty_cblk_builder<Codom>();
    constexpr auto new_builder = hmean_fn.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;

    constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    constexpr auto canon_cb = new_builder.cb;
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 1);

    constexpr mat_traits<Matrix<double, 3, 3>> F_mat;
    constexpr auto F = F_mat(0.0, 0.0, 1.0 / linalg::constexpr_sqrt(2.0), 1.0,
                             0.0, -0.5, 0.0, 1.0, -0.5);
    STATIC_REQUIRE(all_equal(F, canon_cb.map.linear_part()));
    STATIC_REQUIRE(canon_cb.map.translation() == Vectord<3>{});

    constexpr auto hmean_concave = hmean_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(!hmean_concave(Vectord<2>{}).has_value());
    STATIC_REQUIRE(hmean_concave(Vectord<2>{1.0, 1.0}).has_value());
    STATIC_REQUIRE(hmean_concave(Vectord<2>{1.0, 1.0}).value() == -1.0);

    constexpr auto new_builder1 =
        hmean_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = new_builder1.coord_map;
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    constexpr auto canon_cb1 = new_builder1.cb;
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb1.cones)>::value == 1);

    constexpr auto F1 = F_mat(0.0, 0.0, -1.0 / linalg::constexpr_sqrt(2.0), 1.0,
                              0.0, 0.5, 0.0, 1.0, 0.5);
    STATIC_REQUIRE(all_equal(F1, canon_cb1.map.linear_part()));
    STATIC_REQUIRE(canon_cb.map.translation() == Vectord<3>{});
  }
}

TEST_CASE("KL Div Function", "[convex_fn]") {
  SECTION("Test kl div") {
    using Dom = Matrixd<1, 3>;
    using Codom = Matrixd<1, 1>;

    constexpr auto ones = Vectord<3>{1.0, 1.0, 1.0};
    constexpr auto q = Variable<PosVarMeta<Dom>>();

    constexpr auto kldiv_fn =
        KLDivergence<KLDivMeta<Dom, Codom>, decltype(q)>(ones, q);
    STATIC_REQUIRE(!kldiv_fn(Vectord<3>{}).has_value());
    STATIC_REQUIRE(kldiv_fn(ones).has_value());
    STATIC_REQUIRE(kldiv_fn(ones).value() == 0.0);

    constexpr auto builder = create_empty_cblk_builder<Codom>();
    constexpr auto new_builder = kldiv_fn.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;

    constexpr linalg::Vector<double, 10> coord{0.0, 0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    constexpr auto canon_cb = new_builder.cb;
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 2);

    constexpr mat_traits<Matrix<double, 8, 10>> F_mat;
    constexpr auto F = F_mat(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 1.0);
    STATIC_REQUIRE(all_equal(F, canon_cb.map.linear_part()));
    STATIC_REQUIRE(canon_cb.map.translation() == Vectord<8>{});

    constexpr auto kldiv_concave = kldiv_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(!kldiv_concave(Vectord<3>{}).has_value());
    STATIC_REQUIRE(kldiv_concave(ones).has_value());
    STATIC_REQUIRE(kldiv_concave(ones).value() == 0.0);

    constexpr auto new_builder1 =
        kldiv_concave.canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map1 = new_builder1.coord_map;
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map1.linear_part()));

    constexpr auto canon_cb1 = new_builder1.cb;
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb1.cones)>::value == 2);
    constexpr auto F1 = F_mat(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 1.0);
    STATIC_REQUIRE(all_equal(F1, canon_cb1.map.linear_part()));
    STATIC_REQUIRE(canon_cb1.map.translation() == Vectord<8>{});
  }
}
