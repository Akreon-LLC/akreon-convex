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
using namespace unit_test;

TEST_CASE("Test abs function", "[convex_fn]") {
  SECTION("Test abs function") {
    using Space = Matrixd<1, 1>;
    constexpr auto abs_fn = AbsFn<Space>();
    STATIC_REQUIRE(ConvexFnLike<decltype(abs_fn)>);
    STATIC_REQUIRE(abs_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(abs_fn(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(abs_fn(Vectord<1>{-2.0}).has_value());
    STATIC_REQUIRE(abs_fn(Vectord<1>{-2.0}).value() == 2.0);

#if 0
    constexpr auto epi_f = abs_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<1>{0.0}, 0.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<1>{-1.0}, 1.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<1>{-1.0}, 0.0)));
#endif

    constexpr auto abs_concave = abs_fn.negate();
    STATIC_REQUIRE(abs_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(abs_concave(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(abs_concave(Vectord<1>{-2.0}).has_value());
    STATIC_REQUIRE(abs_concave(Vectord<1>{-2.0}).value() == -2.0);

#if 0
    constexpr auto hypo_f = abs_concave.hypograph();
    // ||x||_1 <= -t
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<1>{0.0}, 0.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<1>{1.0}, 1.0)));
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<1>{1.0}, -1.0)));
#endif
  }
}

TEST_CASE("Test affine fn", "[convex_fn]") {
  SECTION("Test affine fn (trivial)") {
    using Space = Matrixd<1, 2>;
    constexpr auto x0 = Vectord<1>{};
    constexpr auto U = Space{};
    constexpr auto map = AffineMap<Space>(U, x0);
    constexpr auto affine_fn = AffineFn<Space>(map);
    STATIC_REQUIRE(affine_fn.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(affine_fn.eval(Vectord<2>{}).value() == 0.0);

    constexpr auto affine_concave = affine_fn.negate();
    STATIC_REQUIRE(ConvexFnLike<decltype(affine_concave.negate())>);

#if 0
    constexpr auto epi_f = affine_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, -1.0)));

    constexpr auto hypo_f = affine_concave.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, -1.0)));
#endif
  }

  SECTION("Test affine fn (convex)") {
    using Space = Matrixd<1, 3>;
    constexpr auto x0 = Vectord<1>{2.0};
    constexpr auto U = Space{Vectord<3>{1.0, 5.0, 2.0}};
    constexpr auto map = AffineMap<Space>(U, x0);
    constexpr auto affine_fn = AffineFn<Space>(map);
    STATIC_REQUIRE(affine_fn.eval(Vectord<3>{}).has_value());
    STATIC_REQUIRE(affine_fn.eval(Vectord<3>{}).value() == 2.0);

#if 0
    constexpr auto epi_f = affine_fn.epigraph();
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<3>{}, 0.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<3>{0.0, 0.0, 0.0}, 2.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<3>{1.0, 1.0, 1.0}, 16.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<3>{1.0, 1.0, 1.0}, 4.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<3>{1.0, 1.0, 0.0}, 1.0)));
#endif
  }

  SECTION("Test affine fn (concave)") {
    using Space = Matrixd<1, 4>;
    constexpr auto x0 = Vectord<1>{1.0};
    constexpr auto U = Space{Vectord<4>{-1.0, 2.0, 3.0, 1.0}};
    constexpr auto map = AffineMap<Space>(U, x0);
    constexpr auto affine_fn = AffineFn<Space, FnTag::Concave>(map);
    STATIC_REQUIRE(affine_fn.eval(Vectord<4>{}).has_value());
    STATIC_REQUIRE(affine_fn.eval(Vectord<4>{}).value() == 1.0);

#if 0
    constexpr auto hypo_f = affine_fn.hypograph();
    // 0 <= 0 T
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<4>{}, 1.0)));
    // [1 -2 -3 -1] [1; -1; -1; -1] = 5 <= 0 ? F
    STATIC_REQUIRE(
        !hypo_f.contains(std::tuple(Vectord<4>{1.0, -1.0, -1.0, -1.0}, 1.0)));
#endif
  }
}

TEST_CASE("Test exp fn", "[convex_fn]") {
  SECTION("Test exp fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto exp_fn = ExpFn<Space>();
    STATIC_REQUIRE(ConvexFnLike<decltype(exp_fn)>);
    STATIC_REQUIRE(exp_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(exp_fn(Vectord<1>{0.0}).value() == 1.0);

#if 0
    constexpr auto epi_f = exp_fn.epigraph();
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<1>{0.0}, 0.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<1>{0.0}, 1.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<1>{1.0}, 3.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<1>{1.0}, 1.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<1>{0.0}, -1.0)));
#endif

    constexpr auto exp_concave = exp_fn.negate();
    STATIC_REQUIRE(exp_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(exp_concave(Vectord<1>{0.0}).value() == -1.0);

#if 0
    constexpr auto hypo_f = exp_concave.hypograph();
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<1>{0.0}, 0.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<1>{0.0}, 1.0)));
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<1>{0.0}, -1.0)));
#endif
  }
}

TEST_CASE("Test geometric mean fn", "[concave_fn]") {
  SECTION("Test geometric mean fn") {
    using Space = Matrixd<1, 4>;
    constexpr auto geo_mean = GeoMeanFn<Space>();
    STATIC_REQUIRE(ConcaveFnLike<decltype(geo_mean)>);
    STATIC_REQUIRE(geo_mean(Vectord<4>{1.0, 1.0, 1.0, 1.0}).has_value());
    STATIC_REQUIRE(geo_mean(Vectord<4>{1.0, 1.0, 1.0, 1.0}).value() == 1.0);
    STATIC_REQUIRE(
        constexpr_abs(geo_mean(Vectord<4>{81.0, 1.0, 1.0, 1.0}).value() -
                      3.0) <= eps);

#if 0
    constexpr auto hypo_f = geo_mean.hypograph();
    STATIC_REQUIRE(
        hypo_f.contains(std::tuple(Vectord<4>{1.0, 1.0, 1.0, 1.0}, 1.0)));
    STATIC_REQUIRE(
        !hypo_f.contains(std::tuple(Vectord<4>{1.0, 1.0, 1.0, 1.0}, 2.0)));
#endif

    constexpr auto geo_mean_convex = geo_mean.negate();
    STATIC_REQUIRE(ConvexFnLike<decltype(geo_mean_convex)>);
    STATIC_REQUIRE(geo_mean_convex(Vectord<4>{1.0, 1.0, 1.0, 1.0}).has_value());
    STATIC_REQUIRE(geo_mean_convex(Vectord<4>{1.0, 1.0, 1.0, 1.0}).value() ==
                   -1.0);

#if 0
    // (dakre) remember this is reflected
    constexpr auto epi_f = geo_mean_convex.epigraph();
    // 0 >= 0 T
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<4>{}, 0.0)));
    // 0 >= -1 T
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<4>{}, 1.0)));
    // 0 >= 1 F
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<4>{}, -1.0)));
#endif
  }
}

TEST_CASE("Test norm fn", "[convex_fn]") {
  SECTION("Test L1 fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, L1<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 3.0);

#if 0
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, 3.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, 1.0)));
#endif

    constexpr auto norm_concave = norm.negate();
    STATIC_REQUIRE(norm_concave(Vectord<2>{1.0, 2.0}).value() == -3.0);
    STATIC_REQUIRE(ConvexFnLike<decltype(norm_concave.negate())>);
#if 0
    constexpr auto hypo_f = norm_concave.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, 3.0)));
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, -3.0)));
#endif
  }

  SECTION("Test L2 fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, L2<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() ==
                   constexpr_sqrt(5.0));

#if 0
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(
        epi_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, constexpr_sqrt(5.0))));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, 1.0)));
#endif

    constexpr auto norm_concave = norm.negate();
    STATIC_REQUIRE(ConvexFnLike<decltype(norm_concave.negate())>);
#if 0
    constexpr auto hypo_f = norm_concave.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(!hypo_f.contains(
        std::tuple(Vectord<2>{1.0, 2.0}, constexpr_sqrt(5.0))));
    STATIC_REQUIRE(hypo_f.contains(
        std::tuple(Vectord<2>{1.0, 2.0}, -1.0 * constexpr_sqrt(5.0))));
#endif
  }

  SECTION("Test Linf fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, Linf<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 2.0);

#if 0
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, 2.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, 1.0)));
#endif

    constexpr auto norm_concave = norm.negate();
    STATIC_REQUIRE(ConvexFnLike<decltype(norm_concave.negate())>);
#if 0
    constexpr auto hypo_f = norm_concave.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{}, 0.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, 2.0)));
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{1.0, 2.0}, -2.0)));
#endif
  }
}

TEST_CASE("Test max fn") {
  SECTION("Test max fn (trivial)") {
    using Space = Matrixd<1, 2>;
    constexpr auto x = Vectord<2>{};
    constexpr auto max_fn = MaxFn<Space>();
    STATIC_REQUIRE(max_fn.eval(x).has_value());
    STATIC_REQUIRE(max_fn.eval(x).value() == 0.0);
#if 0
    constexpr auto epi_f = max_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{}, 0.0)));
#endif
  }

  SECTION("Test max fn") {
    using Space = Matrixd<1, 5>;
    constexpr auto x = Vectord<5>{-1.0, 4.5, 2.0, 3.0, 5.0};
    constexpr auto max_fn = MaxFn<Space>();
    STATIC_REQUIRE(max_fn.eval(x).has_value());
    STATIC_REQUIRE(max_fn.eval(x).value() == 5.0);

#if 0
    constexpr auto epi_f = max_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<5>{}, 1.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(x, 5.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(x, 1.0)));
#endif

    constexpr auto max_concave = max_fn.negate();
    STATIC_REQUIRE(max_concave(x).value() == -5.0);
    STATIC_REQUIRE(ConvexFnLike<decltype(max_concave.negate())>);
#if 0
    constexpr auto hypo_f = max_concave.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<5>{}, 0.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<5>{}, 1.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(x, 5.0)));
#endif
  }
}

TEST_CASE("Test quad over linear", "[convex_fn]") {
  SECTION("Test quad/linear") {
    using Space = Matrixd<1, 4>;
    constexpr auto qol = QuadOverLinearFn<Space>();
    STATIC_REQUIRE(!qol.eval(Vectord<4>{}).has_value());
    STATIC_REQUIRE(qol.eval(Vectord<4>{1.0, 2.0, 3.0, 2.0}).has_value());
    STATIC_REQUIRE(qol.eval(Vectord<4>{1.0, 2.0, 3.0, 2.0}).value() == 7.0);
    STATIC_REQUIRE(!qol.eval(Vectord<4>{1.0, 2.0, 0.0, 0.0}).has_value());

#if 0
    constexpr auto epi_f = qol.epigraph();
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<4>{}, 1.0)));
    STATIC_REQUIRE(
        epi_f.contains(std::tuple(Vectord<4>{1.0, 2.0, 1.0, 2.0}, 3.0)));
#endif

    constexpr auto qol_concave = qol.negate();
    STATIC_REQUIRE(!qol_concave(Vectord<4>{}).has_value());
    STATIC_REQUIRE(qol_concave(Vectord<4>{1.0, 2.0, 3.0, 2.0}).has_value());
    STATIC_REQUIRE(qol_concave(Vectord<4>{1.0, 2.0, 3.0, 2.0}).value() == -7.0);
    STATIC_REQUIRE(ConvexFnLike<decltype(qol_concave.negate())>);
#if 0
    constexpr auto hypo_f = qol_concave.hypograph();
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<4>{}, 1.0)));
    STATIC_REQUIRE(
        !hypo_f.contains(std::tuple(Vectord<4>{1.0, 2.0, 1.0, 2.0}, 3.0)));
    STATIC_REQUIRE(
        hypo_f.contains(std::tuple(Vectord<4>{0.0, 0.0, 0.0, 1.0}, 0.0)));
#endif
  }
}

TEST_CASE("Test log det fn", "[concave_fn]") {
  SECTION("Test log det fn") {
    constexpr auto I = make_identity<double, 3>();
    constexpr auto log_det = LogDetFn<linalg::SymmetricMatrix<decltype(I)>>();
    STATIC_REQUIRE(ConcaveFnLike<decltype(log_det)>);
    STATIC_REQUIRE(log_det(I).has_value());
    STATIC_REQUIRE(log_det(I).value() == 0.0);
    STATIC_REQUIRE(!log_det(-1.0 * I).has_value());

#if 0
    constexpr auto hypo_f = log_det.hypograph();
    // 0 >= 0 T
    STATIC_REQUIRE(hypo_f.contains(std::tuple(I, 0.0)));
    // 0 >= 1 F
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(I, 1.0)));
    // Not PD F
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(-1.0 * I, 1.0)));
#endif

    constexpr auto log_det_convex = log_det.negate();
    STATIC_REQUIRE(log_det_convex(I).has_value());
    STATIC_REQUIRE(log_det_convex(I).value() == 0.0);
    STATIC_REQUIRE(ConvexFnLike<decltype(log_det_convex)>);

#if 0
    // (dakre) remember this is reflected
    constexpr auto epi_f = log_det_convex.epigraph();
    // 0 >= 0 T
    STATIC_REQUIRE(epi_f.contains(std::tuple(I, 0.0)));
    // 0 >= -1 T
    STATIC_REQUIRE(epi_f.contains(std::tuple(I, 1.0)));
    // 0 >= 1 F
    STATIC_REQUIRE(!epi_f.contains(std::tuple(I, -1.0)));
#endif
  }
}

TEST_CASE("Test log sum exp", "[convex_fn]") {
  SECTION("Test log sum exp") {
    using Space = Matrixd<1, 4>;
    constexpr auto lse = LogSumExpFn<Space>();
    STATIC_REQUIRE(lse(Vectord<4>{}).has_value());
    STATIC_REQUIRE(constexpr_abs(lse(Vectord<4>{}).value() - 1.38629436112) <
                   eps);
    STATIC_REQUIRE(lse(Vectord<4>{-1.0, 0.0, 0.0, 0.0}).has_value());
    STATIC_REQUIRE(constexpr_abs(lse(Vectord<4>{-1.0, 0.0, 0.0, 0.0}).value() -
                                 1.21428330036) < eps);

#if 0
    constexpr auto epi_f = lse.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<4>{}, 2.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<4>{}, 1.0)));
#endif

    constexpr auto lse_concave = lse.negate();
    STATIC_REQUIRE(lse_concave(Vectord<4>{}).has_value());
    STATIC_REQUIRE(
        constexpr_abs(lse_concave(Vectord<4>{}).value() + 1.38629436112) < eps);
    STATIC_REQUIRE(ConvexFnLike<decltype(lse_concave.negate())>);

#if 0
    constexpr auto hypo_f = lse_concave.hypograph();
    // log(4 * exp(2)) = 1.57ish
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<4>{}, 2.0)));
    // log(4 * exp(-2)) = 0.6ish
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<4>{}, -2.0)));
#endif
  }
}

TEST_CASE("Test log function", "[convex_fn]") {
  SECTION("Test log fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto log_fn = LogFn<Space>();
    STATIC_REQUIRE(ConcaveFnLike<decltype(log_fn)>);
    STATIC_REQUIRE(!log_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(log_fn(Vectord<1>{1.0}).value() == 0.0);

#if 0
    constexpr auto hypo_f = log_fn.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<1>{1.0}, 0.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<1>{0.0}, 1.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<1>{1.2}, 1.0)));
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<1>{1.2}, -1.0)));
#endif

    constexpr auto log_convex = log_fn.negate();
    STATIC_REQUIRE(!log_convex(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(log_convex(Vectord<1>{1.0}).value() == 0.0);

#if 0
    constexpr auto epi_f = log_convex.epigraph();
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<1>{1.0}, 0.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<1>{0.0}, 1.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<1>{1.2}, 1.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<1>{-3.0}, 1.0)));
#endif
  }
}

TEST_CASE("Test square function", "[convex_fn]") {
  SECTION("Test square function") {
    using Space = Matrixd<1, 1>;
    constexpr auto square_fn = SquareFn<Space>();
    STATIC_REQUIRE(ConvexFnLike<decltype(square_fn)>);
    STATIC_REQUIRE(square_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(square_fn(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(square_fn(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(square_fn(Vectord<1>{2.0}).value() == 4.0);

#if 0
    constexpr auto epi_f = square_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{0.0, 0.0}, 0.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{1.0, 0.0}, 1.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<2>{2.0, 10.0}, 1.0)));
#endif

    constexpr auto square_concave = square_fn.negate();
    STATIC_REQUIRE(square_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(square_concave(Vectord<1>{0.0}).value() == 0.0);
    STATIC_REQUIRE(square_concave(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(square_concave(Vectord<1>{1.0}).value() == -1.0);

#if 0
    constexpr auto hypo_f = square_concave.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{0.0, 0.0}, 0.0)));
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{1.0, 0.0}, -1.0)));
#endif
  }
}

TEST_CASE("Test inv pos function", "[convex_fn]") {
  SECTION("Test inv pos fn") {
    using Space = Matrixd<1, 1>;
    constexpr auto inv_pos_fn = InvPosFn<Space>();
    STATIC_REQUIRE(ConvexFnLike<decltype(inv_pos_fn)>);
    STATIC_REQUIRE(!inv_pos_fn(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(!inv_pos_fn(Vectord<1>{-1.0}).has_value());
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{1.0}).value() == 1.0);
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(inv_pos_fn(Vectord<1>{2.0}).value() == 0.5);

#if 0
    constexpr auto epi_f = inv_pos_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{0.0, 1.0}, 1.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{10.0, 2.0}, 0.5)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<2>{10.0, -2.0}, 0.5)));
#endif

    constexpr auto inv_pos_concave = inv_pos_fn.negate();
    STATIC_REQUIRE(!inv_pos_concave(Vectord<1>{0.0}).has_value());
    STATIC_REQUIRE(!inv_pos_concave(Vectord<1>{-1.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{1.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{1.0}).value() == -1.0);
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{2.0}).has_value());
    STATIC_REQUIRE(inv_pos_concave(Vectord<1>{2.0}).value() == -0.5);

#if 0
    constexpr auto hypo_f = inv_pos_concave.hypograph();
    STATIC_REQUIRE(hypo_f.contains(std::tuple(Vectord<2>{0.0, 1.0}, -1.0)));
    STATIC_REQUIRE(!hypo_f.contains(std::tuple(Vectord<2>{0.0, 1.0}, 1.0)));
#endif
  }
}
