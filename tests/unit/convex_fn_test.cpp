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
    constexpr auto new_builder = abs_fn.canonicalize(builder);
    constexpr auto canon_map = new_builder.coord_map;
    constexpr auto canon_cb = new_builder.cb;

    constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
    STATIC_REQUIRE(all_equal(as_row_vector(coord), canon_map.linear_part()));

    STATIC_REQUIRE(canon_cb.map == affine::AffineMap(F, Vectord<5>{}));

    constexpr auto cones = canon_cb.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 5);
    STATIC_REQUIRE(canon_cb.validate());

    constexpr auto abs_concave = abs_fn.negate();
    STATIC_REQUIRE(abs_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(abs_concave(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(abs_concave(Vectord<1>{-2.0}).has_value());
    STATIC_REQUIRE(abs_concave(Vectord<1>{-2.0}).value() == -2.0);

    constexpr mat_traits<Matrix<double, 5, 3>> F1_mat;
    constexpr auto F1 = F1_mat(1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, -1.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, -1.0);

    constexpr auto builder1 = abs_concave.canonicalize(builder);
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

/*
TEST_CASE("Test exp fn", "[convex_fn]") {
  SECTION("Test exp fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<FreeVarMeta<Space>>();
    constexpr auto exp_fn = Exp<ExpMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(exp_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(exp_fn(Vectord<1>{0.0}).value() == 1.0);

#if 0
    constexpr auto cmap_builder = ConeMapBuilder<Space>();
    constexpr auto cb = create_empty_cblk<double>();
    constexpr auto map_id = affine::make_map_id<Space>();
    constexpr auto canon = exp_fn.canonicalize(cmap_builder, map_id, cb);

    constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    STATIC_REQUIRE(cone_block.map ==
                   affine::AffineMap(F, Vectord<3>{0.0, 1.0, 0.0}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto exp_concave = exp_fn.negate();
    STATIC_REQUIRE(exp_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(exp_concave(Vectord<1>{0.0}).value() == -1.0);

#if 0
    constexpr auto F1 = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, -1.0);

    constexpr auto canon1 = exp_concave.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map ==
                   affine::AffineMap(F1, Vectord<3>{0.0, 1.0, 0.0}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(cone_block1.validate());
#endif
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

#if 0
    constexpr auto cmap_builder = ConeMapBuilder<Space>();
    constexpr auto map_id = affine::make_map_id<linalg::Matrixd<4, 4>>();
    constexpr auto cb = create_empty_cblk<double>();

    constexpr auto F = make_identity<double, 5>();
    constexpr auto canon = geo_mean.canonicalize(cmap_builder, map_id, cb);

    constexpr linalg::Vector<double, 5> coord{0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, Vectord<5>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto geo_mean_convex = geo_mean.negate();
    STATIC_REQUIRE(geo_mean_convex(Vectord<4>{1.0, 1.0, 1.0, 1.0}).has_value());
    STATIC_REQUIRE(geo_mean_convex(Vectord<4>{1.0, 1.0, 1.0, 1.0}).value() ==
                   -1.0);

#if 0
    constexpr mat_traits<Matrix<double, 5, 5>> F_mat;
    constexpr auto F1 =
        F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0);

    constexpr auto canon1 =
        geo_mean_convex.canonicalize(cmap_builder, map_id, cb);

    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map == affine::AffineMap(F1, Vectord<5>{}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(cone_block1.validate());
#endif
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

#if 0
    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(0.0, 0.0, 1.0, 0.0, 0.0, 1.0);

    constexpr auto cmap_builder = ConeMapBuilder<Space>();
    constexpr auto map_id = affine::make_map_id<Space>();
    constexpr auto cb = create_empty_cblk<double>();

    constexpr auto canon = inv_pos_fn.canonicalize(cmap_builder, map_id, cb);

    constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(
        cone_block.map ==
        affine::AffineMap(F, Vectord<3>{constexpr_sqrt(2.0), 0.0, 0.0}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto inv_pos_concave = inv_pos_fn.negate();
    STATIC_REQUIRE(!inv_pos_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(!inv_pos_concave(Vectord<1>{-1.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{1.0}).value() == -1.0);
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{2.0}).value() == -0.5);

#if 0
    constexpr mat_traits<Matrix<double, 3, 2>> F1_mat;
    constexpr auto F1 = F1_mat(0.0, 0.0, 1.0, 0.0, 0.0, -1.0);

    constexpr auto canon1 =
        inv_pos_concave.canonicalize(cmap_builder, map_id, cb);

    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(
        cone_block1.map ==
        affine::AffineMap(F1, Vectord<3>{constexpr_sqrt(2.0), 0.0, 0.0}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(cone_block1.validate());
#endif
  }
}

TEST_CASE("Test log function", "[convex_fn]") {
  SECTION("Test log fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto x = Variable<PosVarMeta<Space>>();
    constexpr auto log_fn = Log<LogMeta<Space>, decltype(x)>(x);
    STATIC_REQUIRE(!log_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(log_fn(Vectord<1>{1.0}).value() == 0.0);

#if 0
    constexpr auto cmap_builder = ConeMapBuilder<Space>();
    constexpr auto map_id = affine::make_map_id<Space>();
    constexpr auto cb = create_empty_cblk<double>();

    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(0.0, 1.0, 0.0, 0.0, 1.0, 0.0);

    constexpr auto canon = log_fn.canonicalize(cmap_builder, map_id, cb);

    constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map ==
                   affine::AffineMap(F, Vectord<3>{0.0, 1.0, 0.0}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto log_convex = log_fn.negate();
    STATIC_REQUIRE(!log_convex(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(log_convex(Vectord<1>{1.0}).value() == 0.0);

#if 0
    constexpr auto F1 = F_mat(0.0, -1.0, 0.0, 0.0, 1.0, 0.0);

    constexpr auto canon1 = log_convex.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map ==
                   affine::AffineMap(F1, Vectord<3>{0.0, 1.0, 0.0}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(cone_block1.validate());
#endif
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

#if 0
    constexpr mat_traits<Matrix<double, 5, 6>> F_mat;
    constexpr auto F =
        F_mat(-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, -1.0, 1.0);

    constexpr auto cmap_builder = ConeMapBuilder<Space>();
    constexpr auto map_id = affine::make_map_id<linalg::Matrix<double, 5, 5>>();
    constexpr auto cb = create_empty_cblk<double>();

    constexpr auto canon = max_fn.canonicalize(cmap_builder, map_id, cb);
    constexpr linalg::Vector<double, 6> coord{0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, Vectord<5>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 5);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto max_concave = max_fn.negate();
    STATIC_REQUIRE(max_concave(x).value() == -5.0);

#if 0
    constexpr auto F1 =
        F_mat(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0,
              0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0,
              0.0, 0.0, 0.0, 0.0, -1.0, -1.0);

    constexpr auto canon1 = max_concave.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map == affine::AffineMap(F1, Vectord<5>{}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 5);
    STATIC_REQUIRE(cone_block1.validate());
#endif
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

#if 0
    constexpr mat_traits<Matrix<double, 3, 3>> F_mat;
    constexpr auto F = F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    constexpr auto canon = norm.canonicalize(cmap_builder, map_id, cb);

    constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, Vectord<3>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());

    constexpr auto norm_concave = norm.negate();
    STATIC_REQUIRE(ConvexFnLike<decltype(norm_concave.negate())>);

    constexpr auto F1 = F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
    constexpr auto canon1 = norm_concave.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map == affine::AffineMap(F1, Vectord<3>{}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(cone_block1.validate());
#endif
  }

  SECTION("Test Linf fn") {
    constexpr auto norm =
        Norm<NormMeta<Space, Codom>, decltype(var), Linf<Vectord<2>>>(var);
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 2.0);

#if 0
    constexpr mat_traits<Matrix<double, 4, 3>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0);

    constexpr auto canon = norm.canonicalize(cmap_builder, map_id, cb);
    constexpr linalg::Vector<double, 3> coord{0.0, 0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, Vectord<4>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());

    constexpr auto norm_concave = norm.negate();
    STATIC_REQUIRE(ConvexFnLike<decltype(norm_concave.negate())>);

    constexpr auto F1 =
        F_mat(1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0);
    constexpr auto canon1 = norm_concave.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map == affine::AffineMap(F1, Vectord<4>{}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 2);
    STATIC_REQUIRE(cone_block1.validate());
#endif
  }

  SECTION("Test L1 fn") {
    constexpr auto norm =
        Norm<NormMeta<Space, Codom>, decltype(var), L1<Vectord<2>>>(var);
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 3.0);

#if 0
    constexpr mat_traits<Matrix<double, 5, 5>> F_mat;
    constexpr auto F =
        F_mat(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 1.0);

    constexpr auto canon = norm.canonicalize(cmap_builder, map_id, cb);
    constexpr linalg::Vector<double, 5> coord{0.0, 0.0, 0.0, 0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, Vectord<5>{}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 3);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto norm_concave = norm.negate();
    STATIC_REQUIRE(norm_concave(Vectord<2>{1.0, 2.0}).value() == -3.0);

#if 0
    constexpr auto F1 =
        F_mat(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0,
              0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0);
    constexpr auto canon1 = norm_concave.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map == affine::AffineMap(F1, Vectord<5>{}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 3);
    STATIC_REQUIRE(cone_block1.validate());
#endif
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

#if 0
    constexpr mat_traits<Matrix<double, 3, 2>> F_mat;
    constexpr auto F = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    constexpr auto cmap_builder = ConeMapBuilder<Space>();
    constexpr auto map_id = affine::make_map_id<Space>();
    constexpr auto cb = create_empty_cblk<double>();

    constexpr auto canon = square_fn.canonicalize(cmap_builder, map_id, cb);

    constexpr linalg::Vector<double, 2> coord{0.0, 1.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map ==
                   affine::AffineMap(F, Vectord<3>{0.0, 0.5, 0.0}));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 1);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto square_concave = square_fn.negate();
    STATIC_REQUIRE(square_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(square_concave(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(square_concave(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(square_concave(Vectord<1>{1.0}).value() == -1.0);

#if 0
    constexpr auto F1 = F_mat(1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    constexpr auto canon1 =
        square_concave.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map ==
                   affine::AffineMap(F1, Vectord<3>{0.0, 0.5, 0.0}));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 1);
    STATIC_REQUIRE(cone_block1.validate());
#endif
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

#if 0
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

    constexpr auto cmap_builder = ConeMapBuilder<Space>();
    constexpr auto map_id = affine::make_map_id<linalg::Matrix<double, 4, 4>>();
    constexpr auto cb = create_empty_cblk<double>();

    constexpr auto canon = lse.canonicalize(cmap_builder, map_id, cb);
    constexpr linalg::Vector<double, 9> coord{0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 1.0, 0.0};
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon).linear_part()));

    constexpr auto cone_block = std::get<1>(canon);
    STATIC_REQUIRE(cone_block.map == affine::AffineMap(F, g));

    constexpr auto cones = cone_block.cones;
    constexpr auto sz = std::tuple_size<decltype(cones)>::value;
    STATIC_REQUIRE(sz == 2);
    STATIC_REQUIRE(cone_block.validate());
#endif

    constexpr auto lse_concave = lse.negate();
    STATIC_REQUIRE(lse_concave(Vectord<4>{}).has_value());
    STATIC_REQUIRE(
        constexpr_abs(lse_concave(Vectord<4>{}).value() + 1.38629436112) < eps);

#if 0
    constexpr auto F1 =
        F_mat(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0);

    constexpr auto canon1 = lse_concave.canonicalize(cmap_builder, map_id, cb);
    STATIC_REQUIRE(
        all_equal(as_row_vector(coord), std::get<0>(canon1).linear_part()));

    constexpr auto cone_block1 = std::get<1>(canon1);
    STATIC_REQUIRE(cone_block1.map == affine::AffineMap(F1, g));

    constexpr auto cones1 = cone_block1.cones;
    constexpr auto sz1 = std::tuple_size<decltype(cones1)>::value;
    STATIC_REQUIRE(sz1 == 2);
    STATIC_REQUIRE(cone_block1.validate());
#endif
  }
}
*/
