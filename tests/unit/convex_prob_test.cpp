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
#include "test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace affine;
using namespace conic;
using namespace unit_test;

TEST_CASE("Test var setup", "[convex_vars]") {
  SECTION("Free var") {
    using MatT = Matrixd<1, 1>;
    constexpr auto x = Variable<VarMeta<MatT>>();
    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(x)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(dom_vartype_v<meta_codom_t<atom_meta_t<decltype(x)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(x)>> ==
                   CurvatureType::AFFINE);
    STATIC_REQUIRE(is_var_type_v<decltype(x)>);

    constexpr auto mat = mat_traits<MatT>();
    constexpr auto four_x = x.mul(mat(4.0));
    STATIC_REQUIRE(four_x.map.linear_part() == mat(4.0));

    constexpr auto four_x_plus_two = four_x.add(Vectord<1>{2.0});
    STATIC_REQUIRE(four_x_plus_two.map.translation() == Vectord<1>{2.0});

    constexpr auto builder = create_empty_cblk_builder<MatT>();
    constexpr auto new_builder =
        four_x_plus_two.template canonicalize<AtomType::VAR>(builder);
    constexpr auto canon1_map = new_builder.coord_map;
    constexpr auto canon1_cb = new_builder.cb;

    constexpr auto noop_var_block =
        four_x_plus_two.template canonicalize<AtomType::VAR>(builder);
    STATIC_REQUIRE(canon1_map == noop_var_block.coord_map);
    STATIC_REQUIRE(canon1_cb.map == noop_var_block.cb.map);
    STATIC_REQUIRE(canon1_cb.cones == noop_var_block.cb.cones);

    STATIC_REQUIRE(canon1_map.linear_part() == mat(4.0));
    STATIC_REQUIRE(canon1_map.translation() == Vectord<1>{2.0});
    STATIC_REQUIRE(std::tuple_size<decltype(canon1_cb.cones)>::value == 0);

    constexpr auto x0 = x(0);
    STATIC_REQUIRE(is_index_type_v<decltype(x0)>);
    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(x0)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(dom_vartype_v<meta_codom_t<atom_meta_t<decltype(x0)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(x0)>> ==
                   CurvatureType::AFFINE);

    STATIC_REQUIRE(std::tuple_size<decltype(x0.children)>::value == 1);
    constexpr auto two_x0 = x0.mul(2.0);
    STATIC_REQUIRE(two_x0.map.linear_part() == mat(2.0));

    constexpr auto two_x0_plus_one = two_x0.add(1.0);
    STATIC_REQUIRE(two_x0_plus_one.map.linear_part() == mat(2.0));
    STATIC_REQUIRE(two_x0_plus_one.map.translation() == Vectord<1>{1.0});

    constexpr auto new_builder2 =
        two_x0_plus_one.template canonicalize<AtomType::VAR>(builder);
    constexpr auto canon2_map = new_builder2.coord_map;
    constexpr auto canon2_cb = new_builder2.cb;
    STATIC_REQUIRE(canon2_map.linear_part() == mat(2.0));
    STATIC_REQUIRE(canon2_map.translation() == Vectord<1>{1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(canon2_cb.cones)>::value == 0);

    constexpr auto sum = Sum(x0, x0);
    STATIC_REQUIRE(std::tuple_size<decltype(sum.children)>::value == 2);

    constexpr auto new_builder3 =
        sum.template canonicalize<AtomType::VAR>(builder);
    constexpr auto canon3_map = new_builder3.coord_map;
    constexpr auto canon3_cb = new_builder3.cb;

    STATIC_REQUIRE(all_equal(canon3_map.linear_part(), mat(2.0)));
    STATIC_REQUIRE(canon3_map.translation() == Vectord<1>{0.0});
    STATIC_REQUIRE(std::tuple_size<decltype(canon3_cb.cones)>::value == 0);
  }

  SECTION("N-dim var") {
    using MatT = Matrixd<1, 5>;
    constexpr auto mat = mat_traits<MatT>();

    constexpr auto x = Variable<VarMeta<MatT>>();
    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(x)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(dom_vartype_v<meta_codom_t<atom_meta_t<decltype(x)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(x)>> ==
                   CurvatureType::AFFINE);
    STATIC_REQUIRE(is_var_type_v<decltype(x)>);

    constexpr auto builder = create_empty_cblk_builder<MatT>();
    constexpr auto new_builder =
        x.template canonicalize<AtomType::VAR>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    STATIC_REQUIRE(canon_map == affine::make_map_id<Matrixd<5, 5>>());

    constexpr auto canon_cb = new_builder.cb;
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 0);

    constexpr auto three_x = x.mul(3.0);
    constexpr auto three_I = 3.0 * make_identity<double, 5>();
    STATIC_REQUIRE(three_x.map.linear_part() == three_I);
    STATIC_REQUIRE(three_x.map.translation() == Vectord<5>{});

    constexpr auto x3 = three_x(3);
    STATIC_REQUIRE(is_index_type_v<decltype(x3)>);
    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(x3)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(dom_vartype_v<meta_codom_t<atom_meta_t<decltype(x3)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(x3)>> ==
                   CurvatureType::AFFINE);

    constexpr auto two_x3 = x3.mul(2.0);
    STATIC_REQUIRE(
        all_equal(two_x3.map.linear_part(), mat(0.0, 0.0, 0.0, 6.0, 0.0)));

    constexpr auto two_x3_plus_one = two_x3.add(2.0);
    STATIC_REQUIRE(two_x3_plus_one.map.linear_part() ==
                   mat(0.0, 0.0, 0.0, 6.0, 0.0));
    STATIC_REQUIRE(two_x3_plus_one.map.translation() == Vectord<1>{2.0});
  }

  SECTION("Var -> Scalar View -> Abs") {
    using Space = Matrixd<1, 3>;
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto x1 = x(1).mul(1.5).add(-1.0);
    constexpr auto abs_fn = Abs(x1);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder =
        abs_fn.template canonicalize<AtomType::FN>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr auto mat = mat_traits<Matrixd<1, 5>>();
    STATIC_REQUIRE(canon_map.linear_part() == mat(0.0, 0.0, 0.0, 0.0, 1.0));
    STATIC_REQUIRE(canon_map.translation() == Vectord<1>{0.0});

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 5);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto map = canon_cb.map;
    STATIC_REQUIRE(mat_rows_v<decltype(map.linear_part())> == 5);
    STATIC_REQUIRE(mat_cols_v<decltype(map.linear_part())> == 5);
    STATIC_REQUIRE(vec_size_v<decltype(map.translation())> == 5);
    constexpr mat_traits<Matrix<double, 5, 5>> F_mat;
    constexpr auto F =
        F_mat(0.0, 1.5, 0.0, 1.0, 0.0, 0.0, -1.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    STATIC_REQUIRE(all_equal(map.linear_part(), F));
    STATIC_REQUIRE(map.translation() == Vectord<5>{-1.0, 1.0, 0.0, 0.0, 0.0});
  }

  SECTION("Var -> cones") {
    using Space = Matrixd<1, 5>;
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto builder2 = x.add_var_block<AtomType::VAR>(builder);
    constexpr auto I = linalg::make_identity<double, 5>();
    STATIC_REQUIRE(std::tuple_size<decltype(builder2.cb.cones)>::value == 0);
    STATIC_REQUIRE(builder2.cb.map.linear_part() == Matrixd<0, 0>{});
    STATIC_REQUIRE(builder2.coord_map.linear_part() == I);
  }

  SECTION("PSD var -> cones") {
    using Mat = Matrixd<2, 2>;
    constexpr auto X = Variable<PsdVarMeta<Mat>>();
    using Space = dom_space_t<meta_dom_t<atom_meta_t<decltype(X)>>>;
    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto builder2 = X.add_var_block<AtomType::VAR>(builder);

    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(X)>>> ==
                   VarType::PSD);
    STATIC_REQUIRE(dom_vartype_v<meta_codom_t<atom_meta_t<decltype(X)>>> ==
                   VarType::PSD);
    STATIC_REQUIRE(dom_size_v<meta_dom_t<atom_meta_t<decltype(X)>>> == 3);
    STATIC_REQUIRE(std::same_as<dom_point_t<meta_dom_t<atom_meta_t<decltype(X)>>>,
                                Mat>);

    using ConeT = std::tuple_element_t<0, decltype(builder2.cb.cones)>;
    STATIC_REQUIRE(std::tuple_size<decltype(builder2.cb.cones)>::value == 1);
    STATIC_REQUIRE(std::same_as<ConeT,
                                PositiveSemidefiniteCone<linalg::Vector<double, 3>, 2>>);
    STATIC_REQUIRE(cone_dim_t<ConeT> == 3);
    STATIC_REQUIRE(builder2.cb.validate());

    constexpr auto I = linalg::make_identity<double, 3>();
    STATIC_REQUIRE(builder2.coord_map.linear_part() == I);
    STATIC_REQUIRE(builder2.coord_map.translation() == Vectord<3>{});
    STATIC_REQUIRE(builder2.cb.map.linear_part() == I);
    STATIC_REQUIRE(builder2.cb.map.translation() == Vectord<3>{});
  }

  SECTION("Var + Fn -> Cones") {
    using Space = Matrixd<1, 2>;
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto builder2 = x.add_var_block<AtomType::VAR>(builder);
    constexpr auto I = linalg::make_identity<double, 2>();
    STATIC_REQUIRE(std::tuple_size<decltype(builder2.cb.cones)>::value == 0);
    STATIC_REQUIRE(builder2.coord_map.linear_part() == I);
    STATIC_REQUIRE(builder2.coord_map.translation() == Vectord<2>{});

    constexpr auto max_fn = Max(x);
    constexpr auto builder3 =
        max_fn.template canonicalize<AtomType::FN>(builder2);
    STATIC_REQUIRE(std::tuple_size<decltype(builder3.cb.cones)>::value == 1);
    STATIC_REQUIRE(mat_rows_v<decltype(builder3.cb.map.linear_part())> == 2);
    STATIC_REQUIRE(mat_cols_v<decltype(builder3.cb.map.linear_part())> == 3);
    STATIC_REQUIRE(vec_size_v<decltype(builder3.cb.map.translation())> == 2);

    constexpr mat_traits<Matrix<double, 2, 3>> F_mat;
    constexpr auto F = F_mat(-1.0, 0.0, 1.0, 0.0, -1.0, 1.0);
    STATIC_REQUIRE(all_equal(F, builder3.cb.map.linear_part()));
    STATIC_REQUIRE(builder3.cb.map.translation() == Vectord<2>{});
  }

  SECTION("Split var") {
    using Space = Matrixd<1, 3>;
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto x_tup = x.split();
    STATIC_REQUIRE(std::tuple_size<decltype(x_tup)>::value == 3);

    constexpr auto x0 = std::get<0>(x_tup);
    constexpr auto x1 = std::get<1>(x_tup);
    constexpr auto x2 = std::get<2>(x_tup);

    constexpr auto sel = mat_traits<Space>();
    STATIC_REQUIRE(all_equal(x0.map.linear_part(), sel(1.0, 0.0, 0.0)));
    STATIC_REQUIRE(all_equal(x1.map.linear_part(), sel(0.0, 1.0, 0.0)));
    STATIC_REQUIRE(all_equal(x2.map.linear_part(), sel(0.0, 0.0, 1.0)));
    STATIC_REQUIRE(x0.map.translation() == Vectord<1>{0.0});
    STATIC_REQUIRE(x1.map.translation() == Vectord<1>{0.0});
    STATIC_REQUIRE(x2.map.translation() == Vectord<1>{0.0});

    constexpr auto abs_x0 = Abs(x0);
    constexpr auto abs_x1 = Abs(x1);
    constexpr auto abs_x2 = Abs(x2);
    constexpr auto sum01 = Sum(abs_x0, abs_x1);
    constexpr auto sum012 = Sum(sum01, abs_x2);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder =
        sum012.template canonicalize<AtomType::ADD>(builder);

    constexpr auto canon_map = new_builder.coord_map;
    constexpr linalg::Vector<double, 9> coord{0.0, 0.0, 0.0, 0.0, 1.0,
                                              0.0, 1.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));
    STATIC_REQUIRE(canon_map.translation() == Vectord<1>{0.0});

    constexpr auto canon_cb = new_builder.cb;
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 15);
    STATIC_REQUIRE(canon_cb.validate());
    STATIC_REQUIRE(mat_rows_v<decltype(canon_cb.map.linear_part())> == 15);
    STATIC_REQUIRE(mat_cols_v<decltype(canon_cb.map.linear_part())> == 9);
    STATIC_REQUIRE(vec_size_v<decltype(canon_cb.map.translation())> == 15);
  }

  SECTION("Weighted sum") {
    using MatT = Matrixd<1, 5>;
    constexpr auto x = Variable<VarMeta<MatT>>();
    constexpr Vectord<5> c{1.0, 1.0, 1.0, 1.0, 1.0};
    constexpr auto sum = WeightedSum<MatT>()(c, x);

    constexpr auto builder = create_empty_cblk_builder<MatT>();
    constexpr auto new_builder =
        sum.template canonicalize<AtomType::ADD>(builder);

    constexpr auto canon_map = new_builder.coord_map;
    STATIC_REQUIRE(all_equal(as_row_vector(c), canon_map.linear_part()));
    STATIC_REQUIRE(canon_map.translation() == Vectord<1>{0.0});

    constexpr auto canon_cb = new_builder.cb;
    STATIC_REQUIRE(canon_cb.validate());
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 0);

    constexpr Vectord<5> c2{1.0, 2.0, -1.0, 4.0, 0.0};
    constexpr auto sum2 = WeightedSum<MatT>()(c2, x);

    constexpr auto new_builder2 =
        sum2.template canonicalize<AtomType::ADD>(builder);

    constexpr auto canon_map2 = new_builder2.coord_map;
    STATIC_REQUIRE(all_equal(as_row_vector(c2), canon_map2.linear_part()));
    STATIC_REQUIRE(canon_map2.translation() == Vectord<1>{0.0});

    constexpr auto canon_cb2 = new_builder2.cb;
    STATIC_REQUIRE(canon_cb2.validate());
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb2.cones)>::value == 0);
  }
}

TEST_CASE("Simple function ops", "[convex_ops]") {
  SECTION("Composed abs") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto abs_x = Abs(x);
    // Self-type nesting is intentionally explicit here. `Abs(abs_x)` would
    // trigger CTAD's copy deduction candidate and just copy `abs_x` instead of
    // forming the nested expression `abs(abs(x))`.
    constexpr auto abs_abs_x =
        Abs<AbsMeta<Space>, decltype(abs_x)>(abs_x);

    STATIC_REQUIRE(
        dom_vartype_v<meta_dom_t<atom_meta_t<decltype(abs_abs_x)>>> ==
        VarType::FREE);
    STATIC_REQUIRE(
        dom_vartype_v<meta_codom_t<atom_meta_t<decltype(abs_abs_x)>>> ==
        VarType::NONNEG);
    STATIC_REQUIRE(meta_curv_v<atom_meta_t<decltype(abs_abs_x)>> ==
                   CurvatureType::CONVEX);
    STATIC_REQUIRE(is_fn_type_v<decltype(abs_abs_x)>);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder =
        abs_abs_x.template canonicalize<AtomType::FN>(builder);
    constexpr auto canon_cb = new_builder.cb;

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 10);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto mat = mat_traits<Matrixd<1, 5>>();
    STATIC_REQUIRE(
        all_equal(canon_map.linear_part(), mat(0.0, 0.0, 0.0, 0.0, 1.0)));
    STATIC_REQUIRE(canon_map.translation() == Vectord<1>{});

    constexpr auto map = canon_cb.map;
    STATIC_REQUIRE(mat_rows_v<decltype(map.linear_part())> == 10);
    STATIC_REQUIRE(mat_cols_v<decltype(map.linear_part())> == 5);
    STATIC_REQUIRE(vec_size_v<decltype(map.translation())> == 10);
    constexpr mat_traits<Matrix<double, 10, 5>> F_mat;
    constexpr auto F =
        F_mat(1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0,
              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
              1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    STATIC_REQUIRE(all_equal(map.linear_part(), F));
    STATIC_REQUIRE(map.translation() == Vectord<10>{});
  }

  SECTION("Restricted domain and codomain metadata") {
    using Scalar = Matrixd<1, 1>;
    using Pair = Matrixd<1, 2>;

    constexpr auto x_free = Variable<VarMeta<Scalar>>();
    constexpr auto x_nonneg = Variable<NonnegVarMeta<Scalar>>();
    constexpr auto x_pos = Variable<PosVarMeta<Scalar>>();
    constexpr auto pair_pos = Variable<PosVarMeta<Pair>>();

    constexpr auto exp_fn = Exp(x_free);
    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(exp_fn)>>> ==
                   VarType::FREE);
    STATIC_REQUIRE(dom_vartype_v<meta_codom_t<atom_meta_t<decltype(exp_fn)>>> ==
                   VarType::NONNEG);

    constexpr auto sqrt_fn = Sqrt(x_nonneg);
    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(sqrt_fn)>>> ==
                   VarType::NONNEG);
    STATIC_REQUIRE(
        dom_vartype_v<meta_codom_t<atom_meta_t<decltype(sqrt_fn)>>> ==
        VarType::NONNEG);

    constexpr auto log_fn = Log(x_pos);
    STATIC_REQUIRE(dom_vartype_v<meta_dom_t<atom_meta_t<decltype(log_fn)>>> ==
                   VarType::POS);
    STATIC_REQUIRE(dom_vartype_v<meta_codom_t<atom_meta_t<decltype(log_fn)>>> ==
                   VarType::FREE);

    constexpr auto inv_pos_fn = InvPos(x_pos);
    STATIC_REQUIRE(
        dom_vartype_v<meta_dom_t<atom_meta_t<decltype(inv_pos_fn)>>> ==
        VarType::POS);
    STATIC_REQUIRE(
        dom_vartype_v<meta_codom_t<atom_meta_t<decltype(inv_pos_fn)>>> ==
        VarType::POS);

    constexpr auto geo_mean_fn = GeoMean(pair_pos);
    STATIC_REQUIRE(
        dom_vartype_v<meta_dom_t<atom_meta_t<decltype(geo_mean_fn)>>> ==
        VarType::POS);
    STATIC_REQUIRE(
        dom_vartype_v<meta_codom_t<atom_meta_t<decltype(geo_mean_fn)>>> ==
        VarType::POS);

    constexpr auto harmonic_mean_fn = HarmonicMean(pair_pos);
    STATIC_REQUIRE(
        dom_vartype_v<meta_dom_t<atom_meta_t<decltype(harmonic_mean_fn)>>> ==
        VarType::POS);
    STATIC_REQUIRE(
        dom_vartype_v<meta_codom_t<atom_meta_t<decltype(harmonic_mean_fn)>>> ==
        VarType::POS);

    constexpr auto sqrt_convex = sqrt_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(
        dom_vartype_v<meta_codom_t<atom_meta_t<decltype(sqrt_convex)>>> ==
        VarType::NONPOS);

    constexpr auto geo_mean_convex = geo_mean_fn.mul<-1.0>().add(0.0);
    STATIC_REQUIRE(
        dom_vartype_v<meta_codom_t<atom_meta_t<decltype(geo_mean_convex)>>> ==
        VarType::NEG);
  }

  SECTION("Sum of abs") {
    using MatT = Matrixd<1, 2>;
    constexpr auto x = Variable<VarMeta<MatT>>();
    constexpr auto x0 = x(0);
    constexpr auto x1 = x(1);
    constexpr auto sum = Sum(x0, x1);

    STATIC_REQUIRE(std::tuple_size<decltype(sum.children)>::value == 2);

    constexpr auto builder = create_empty_cblk_builder<MatT>();
    constexpr auto new_builder =
        sum.template canonicalize<AtomType::ADD>(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr auto mat = mat_traits<MatT>();
    STATIC_REQUIRE(all_equal(canon_map.linear_part(), mat(1.0, 1.0)));
    STATIC_REQUIRE(canon_map.translation() == Vectord<1>{0.0});
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 0);

    constexpr auto abs_x0 = Abs(x0);
    constexpr auto abs_x1 = Abs(x1);
    constexpr auto sum_abs = Sum(abs_x0, abs_x1);
    STATIC_REQUIRE(std::tuple_size<decltype(sum_abs.children)>::value == 2);

    constexpr auto new_builder2 =
        sum_abs.template canonicalize<AtomType::ADD>(builder);

    constexpr auto canon2_map = new_builder2.coord_map;

    constexpr linalg::Vector<double, 6> coord{0.0, 0.0, 0.0, 1.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon2_map.linear_part()));

    constexpr auto canon2_cb = new_builder2.cb;
    STATIC_REQUIRE(std::tuple_size<decltype(canon2_cb.cones)>::value == 10);

    constexpr auto map = canon2_cb.map;
    STATIC_REQUIRE(mat_rows_v<decltype(map.linear_part())> == 10);
    STATIC_REQUIRE(mat_cols_v<decltype(map.linear_part())> == 6);
    STATIC_REQUIRE(vec_size_v<decltype(map.translation())> == 10);
    constexpr mat_traits<Matrix<double, 10, 6>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    STATIC_REQUIRE(all_equal(map.linear_part(), F));
    STATIC_REQUIRE(map.translation() == Vectord<10>{});
  }
}

TEST_CASE("Simple convex set ops", "[convex_ops]") {
  SECTION("Intersection of sets") {
    using Space = Matrixd<1, 3>;
    constexpr Vector<double, 3> point{1.0, 2.0, 3.0};
    constexpr auto b = 0.6;
    constexpr auto x = Variable<VarMeta<Space>>();
    constexpr auto hyperplane = Hyperplane(point, b, x);
    STATIC_REQUIRE(hyperplane.is_well_formed());

    constexpr auto r = 2.0;
    constexpr auto ball =
        NormBall<NormBallMeta<Space>, decltype(x), NormBallType::L2>(
            Vectord<3>{}, r, x);
    STATIC_REQUIRE(ball.is_well_formed());

    constexpr auto intersection = Intersection(hyperplane, ball);

    constexpr auto builder = create_empty_cblk_builder<Space>();
    constexpr auto new_builder =
        intersection.template canonicalize<AtomType::CAP>(builder);
    constexpr auto canon_cb = new_builder.cb;
    constexpr auto eq_map = new_builder.eq_map;
    STATIC_REQUIRE(std::tuple_size<decltype(canon_cb.cones)>::value == 1);
    STATIC_REQUIRE(mat_rows_v<decltype(canon_cb.map.linear_part())> == 4);
    STATIC_REQUIRE(mat_cols_v<decltype(canon_cb.map.linear_part())> == 3);
    STATIC_REQUIRE(mat_rows_v<decltype(eq_map.linear_part())> == 1);
    STATIC_REQUIRE(mat_cols_v<decltype(eq_map.linear_part())> == 3);
  }
}

TEST_CASE("Convex Problem Formation", "[convex_problem]") {
  using Dom = Matrixd<1, 3>;
  using Codom = Matrixd<1, 1>;
  constexpr auto var = Variable<VarMeta<Dom>>();
  constexpr auto prob = make_problem(var);

  SECTION("Simple L2 min. problem") {
    constexpr auto norm =
        Norm<NormMeta<Dom, Codom>, decltype(var), L2<Vectord<3>>>(var);
    constexpr auto prob2 = prob.minimize(norm);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob2)>);
    STATIC_REQUIRE(prob2.validate());
    STATIC_REQUIRE(prob2.linear_objective() == Vectord<4>{0.0, 0.0, 0.0, 1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob2.constraint_cones())>::value ==
                   1);
    STATIC_REQUIRE(all_equal(prob2.constraint_map().linear_part(),
                             make_identity<double, 4>()));
    STATIC_REQUIRE(prob2.constraint_map().translation() == Vectord<4>{});
  }

  SECTION("Simple L1 ball constraint min. x(0) prob") {
    constexpr auto c = Vectord<3>{};
    constexpr auto r = 1.5;
    constexpr auto ball =
        NormBall<NormBallMeta<Dom>, decltype(var), NormBallType::L1>(c, r, var);
    STATIC_REQUIRE(ball.is_well_formed());

    constexpr auto prob2 = prob.in_set(ball);
    constexpr auto prob3 = prob2.minimize(var(0));
    STATIC_REQUIRE(is_problem_set_v<decltype(prob3)>);
    STATIC_REQUIRE(prob3.validate());
    STATIC_REQUIRE(prob3.linear_objective() ==
                   Vectord<6>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob3.constraint_cones())>::value ==
                   4);

    constexpr mat_traits<Matrix<double, 10, 6>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
              -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    constexpr linalg::Vector<double, 10> g{0.0, 0.0, 0.0, 0.0, 0.0,
                                           0.0, r,   0.0, 0.0, 0.0};
    STATIC_REQUIRE(all_equal(prob3.constraint_map().linear_part(), F));
    STATIC_REQUIRE(prob3.constraint_map().translation() == g);
  }

  SECTION("Simple EQ constrained min. x(1) prob") {
    constexpr auto I = make_identity<double, 3>();
    constexpr auto subspace = AffineSubspace(I, Vectord<3>{}, var);
    constexpr auto prob2 = prob.in_set(subspace);
    constexpr auto prob3 = prob2.minimize(var(1));
    STATIC_REQUIRE(is_problem_set_v<decltype(prob3)>);
    STATIC_REQUIRE(prob3.validate());
    STATIC_REQUIRE(prob3.linear_objective() == Vectord<3>{0.0, 1.0, 0.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob3.constraint_cones())>::value ==
                   0);
    STATIC_REQUIRE(
        mat_rows_v<decltype(prob3.constraint_map().linear_part())> == 0);
    STATIC_REQUIRE(
        mat_cols_v<decltype(prob3.constraint_map().linear_part())> == 0);
    STATIC_REQUIRE(
        vec_size_v<decltype(prob3.constraint_map().translation())> == 0);
    STATIC_REQUIRE(all_equal(prob3.equality_map().linear_part(), I));
    STATIC_REQUIRE(prob3.equality_map().translation() == Vectord<3>{});
  }

  SECTION("Intersection of sets min. Linf prob") {
    constexpr Vectord<3> a{1.0, 4.0, 6.0};
    constexpr double b = 0.4;
    constexpr auto hyperplane = Hyperplane(a, b, var);
    constexpr Vectord<3> c{1.0, 0.0, 1.0};
    constexpr auto P = make_identity<double, 3>();
    constexpr auto ellipsoid = Ellipsoid(c, P, var);

    constexpr auto intersection = Intersection(hyperplane, ellipsoid);
    constexpr auto prob2 = prob.in_set(intersection);

    constexpr auto norm =
        Norm<NormMeta<Dom, Codom>, decltype(var), Linf<Vectord<3>>>(var);
    constexpr auto prob3 = prob2.minimize(norm);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob3)>);
    STATIC_REQUIRE(prob3.validate());
    STATIC_REQUIRE(prob3.linear_objective() == Vectord<4>{0.0, 0.0, 0.0, 1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob3.constraint_cones())>::value ==
                   3);

    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob3.constraint_map().linear_part())> ==
        10);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob3.constraint_map().linear_part())> ==
        4);
    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob3.equality_map().linear_part())> == 1);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob3.equality_map().linear_part())> == 4);
  }

  SECTION("Sum square min. prob") {
    constexpr auto x0 = var(0);
    constexpr auto x1 = var(1);
    constexpr auto x2 = var(2);
    constexpr auto sq0 = Square(x0);
    constexpr auto sq1 = Square(x1);
    constexpr auto sq2 = Square(x2);
    constexpr auto sum = Sum(sq0, sq1, sq2);
    constexpr auto prob2 = prob.minimize(sum);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob2)>);
    STATIC_REQUIRE(prob2.validate());
    STATIC_REQUIRE(prob2.linear_objective() ==
                   Vectord<6>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob2.constraint_cones())>::value ==
                   3);

    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob2.constraint_map().linear_part())> ==
        9);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob2.constraint_map().linear_part())> ==
        6);
    constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);
    constexpr mat_traits<Matrix<double, 9, 6>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -inv_sqrt2, 0.0, 0.0,
              0.0, 0.0, 0.0, inv_sqrt2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, -inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0,
              inv_sqrt2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, -inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, inv_sqrt2);
    STATIC_REQUIRE(all_equal(prob2.constraint_map().linear_part(), F));
    STATIC_REQUIRE(prob2.constraint_map().translation() ==
                   Vectord<9>{0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2, 0.0,
                              0.5 * inv_sqrt2, 0.5 * inv_sqrt2, 0.0,
                              0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
  }

  SECTION("Simple leq & geq min. prob") {
    constexpr auto x0 = var(0);
    constexpr auto x1 = var(1);
    constexpr auto x2 = var(2);
    constexpr auto exp0 = Exp(x0).template mul<-1.0>();
    constexpr auto exp1 = Exp(x1);
    constexpr auto exp2 = Exp(x2);

    /// (dakre) part 1 - leq
    constexpr auto leq =
        Inequality<InequalityType::LEQ, decltype(exp0), decltype(exp1)>(exp0,
                                                                        exp1);
    constexpr auto prob2 = prob.add_constraint(leq);
    constexpr auto prob3 = prob2.minimize(exp2);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob3)>);
    STATIC_REQUIRE(prob3.validate());

    STATIC_REQUIRE(prob3.linear_objective() ==
                   Vectord<5>{0.0, 0.0, 0.0, 0.0, 1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob3.constraint_cones())>::value ==
                   4);

    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob3.constraint_map().linear_part())> ==
        10);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob3.constraint_map().linear_part())> ==
        5);
    constexpr mat_traits<Matrix<double, 10, 5>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    STATIC_REQUIRE(all_equal(prob3.constraint_map().linear_part(), F));
    STATIC_REQUIRE(
        prob3.constraint_map().translation() ==
        Vectord<10>{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    /// (dakre) part 2 - geq
    constexpr auto geq =
        Inequality<InequalityType::GEQ, decltype(exp0), decltype(exp1)>(exp0,
                                                                        exp1);
    constexpr auto prob4 = prob.add_constraint(geq);
    constexpr auto prob5 = prob4.minimize(exp2);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob5)>);
    STATIC_REQUIRE(prob5.validate());

    STATIC_REQUIRE(prob5.linear_objective() ==
                   Vectord<5>{0.0, 0.0, 0.0, 0.0, 1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob5.constraint_cones())>::value ==
                   4);

    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob5.constraint_map().linear_part())> ==
        10);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob5.constraint_map().linear_part())> ==
        5);
    constexpr auto F1 =
        F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    STATIC_REQUIRE(all_equal(prob5.constraint_map().linear_part(), F1));
    STATIC_REQUIRE(
        prob5.constraint_map().translation() ==
        Vectord<10>{0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  }

  SECTION("Simple eq min. prob") {
    constexpr auto x0 = var(0);
    constexpr auto x1 = var(1);
    constexpr auto x2 = var(2);

    constexpr auto eq =
        Inequality<InequalityType::EQ, decltype(x0), decltype(x1)>(x0, x1);
    constexpr auto prob2 = prob.add_constraint(eq);
    constexpr auto prob3 = prob2.minimize(x2);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob3)>);
    STATIC_REQUIRE(prob3.validate());
    STATIC_REQUIRE(prob3.linear_objective() == Vectord<3>{0.0, 0.0, 1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob3.constraint_cones())>::value ==
                   0);

    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob3.constraint_map().linear_part())> ==
        0);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob3.constraint_map().linear_part())> ==
        0);
    STATIC_REQUIRE(
        vec_size_v<decltype(prob3.constraint_map().translation())> == 0);

    constexpr mat_traits<Matrix<double, 1, 3>> Aeq_mat;
    constexpr auto Aeq = Aeq_mat(1.0, -1.0, 0.0);
    STATIC_REQUIRE(all_equal(prob3.equality_map().linear_part(), Aeq));
    STATIC_REQUIRE(prob3.equality_map().translation() == Vectord<1>{0.0});
  }

  SECTION("Feasibility preserves cone map") {
    constexpr auto c = Vectord<3>{};
    constexpr auto r = 1.5;
    constexpr auto ball =
        NormBall<NormBallMeta<Dom>, decltype(var), NormBallType::L1>(c, r, var);
    constexpr auto prob2 = prob.in_set(ball);
    constexpr auto prob3 = prob2.feasibility();

    STATIC_REQUIRE(is_problem_set_v<decltype(prob3)>);
    STATIC_REQUIRE(prob3.validate());
    STATIC_REQUIRE(prob3.linear_objective() == Vectord<6>{});
    STATIC_REQUIRE(std::tuple_size<decltype(prob3.constraint_cones())>::value ==
                   4);

    constexpr mat_traits<Matrix<double, 10, 6>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
              -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    constexpr linalg::Vector<double, 10> g{0.0, 0.0, 0.0, 0.0, 0.0,
                                           0.0, r,   0.0, 0.0, 0.0};
    STATIC_REQUIRE(all_equal(prob3.constraint_map().linear_part(), F));
    STATIC_REQUIRE(prob3.constraint_map().translation() == g);
  }

  SECTION("Feasibility preserves equality map") {
    constexpr auto I = make_identity<double, 3>();
    constexpr auto subspace = AffineSubspace(I, Vectord<3>{}, var);
    constexpr auto prob2 = prob.in_set(subspace);
    constexpr auto prob3 = prob2.feasibility();

    STATIC_REQUIRE(is_problem_set_v<decltype(prob3)>);
    STATIC_REQUIRE(prob3.validate());
    STATIC_REQUIRE(prob3.linear_objective() == Vectord<3>{});
    STATIC_REQUIRE(std::tuple_size<decltype(prob3.constraint_cones())>::value ==
                   0);
    STATIC_REQUIRE(
        mat_rows_v<decltype(prob3.constraint_map().linear_part())> == 0);
    STATIC_REQUIRE(
        mat_cols_v<decltype(prob3.constraint_map().linear_part())> == 0);
    STATIC_REQUIRE(
        vec_size_v<decltype(prob3.constraint_map().translation())> == 0);
    STATIC_REQUIRE(all_equal(prob3.equality_map().linear_part(), I));
    STATIC_REQUIRE(prob3.equality_map().translation() == Vectord<3>{});
  }
}

TEST_CASE("Scalar Concave Max. Problems", "[convex_problem]") {
  SECTION("Simple log max. problem") {
    using Space = Matrixd<1, 1>;
    constexpr auto var = Variable<PosVarMeta<Space>>();
    constexpr auto prob = make_problem(var);

    constexpr auto log = Log(var);
    constexpr auto prob2 = prob.maximize(log);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob2)>);
    STATIC_REQUIRE(prob2.validate());
    STATIC_REQUIRE(prob2.linear_objective() == Vectord<2>{0.0, 1.0});
    STATIC_REQUIRE(std::tuple_size<decltype(prob2.constraint_cones())>::value ==
                   1);

    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob2.constraint_map().linear_part())> ==
        3);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob2.constraint_map().linear_part())> ==
        2);

    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(0.0, -1.0, 0.0, 0.0, 1.0, 0.0);
    STATIC_REQUIRE(all_equal(prob2.constraint_map().linear_part(), F));
    STATIC_REQUIRE(prob2.constraint_map().translation() ==
                   Vectord<3>{0.0, 1.0, 0.0});
  }

  SECTION("Simple sum sqrt max. problem") {
    using Dom = Matrixd<1, 2>;
    constexpr auto var = Variable<NonnegVarMeta<Dom>>();
    constexpr auto prob = make_problem(var);

    constexpr auto x0 = var(0);
    constexpr auto x1 = var(1);
    constexpr auto sqrt0 = Sqrt(x0);
    constexpr auto sqrt1 = Sqrt(x1);
    constexpr auto sum_sqrt = Sum(sqrt0, sqrt1);

    constexpr auto prob2 = prob.maximize(sum_sqrt);
    STATIC_REQUIRE(is_problem_set_v<decltype(prob2)>);
    STATIC_REQUIRE(prob2.validate());

    STATIC_REQUIRE(std::tuple_size<decltype(prob2.constraint_cones())>::value ==
                   2);
    STATIC_REQUIRE(prob2.linear_objective() ==
                   Vectord<4>{0.0, 0.0, -1.0, -1.0});

    STATIC_REQUIRE(
        linalg::mat_rows_v<decltype(prob2.constraint_map().linear_part())> ==
        6);
    STATIC_REQUIRE(
        linalg::mat_cols_v<decltype(prob2.constraint_map().linear_part())> ==
        4);

    constexpr auto inv_sqrt2 = 1.0 / constexpr_sqrt_safe(2.0);
    constexpr mat_traits<Matrix<double, 6, 4>> F_mat;
    constexpr auto F = F_mat(0.0, 0.0, 1.0, 0.0, -inv_sqrt2, 0.0, 0.0, 0.0,
                             inv_sqrt2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                             -inv_sqrt2, 0.0, 0.0, 0.0, inv_sqrt2, 0.0, 0.0);
    STATIC_REQUIRE(all_equal(prob2.constraint_map().linear_part(), F));
    STATIC_REQUIRE(prob2.constraint_map().translation() ==
                   Vectord<6>{0.0, 0.5 * inv_sqrt2, 0.5 * inv_sqrt2, 0.0,
                              0.5 * inv_sqrt2, 0.5 * inv_sqrt2});
  }
}
