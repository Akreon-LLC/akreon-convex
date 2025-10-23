/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "math.hpp"
#include "unit_test_helpers.hpp"

using namespace linalg;
using namespace convex;
using namespace unit_test;

TEST_CASE("Test norm fn", "[convex_fn]") {
  SECTION("Test L1 fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, L1<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 3.0);
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{1.0, 2.0, 3.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, 1.0}));
  }

  SECTION("Test L2 fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, L2<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() ==
                   constexpr_sqrt(5.0));
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{1.0, 2.0, constexpr_sqrt(5.0)}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, 1.0}));
  }

  SECTION("Test Linf fn") {
    using Space = Matrixd<1, 2>;
    constexpr auto norm = NormFn<Space, Linf<Vectord<2>>>();
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(norm.eval(Vectord<2>{}).value() <= eps);
    STATIC_REQUIRE(norm.eval(Vectord<2>{1.0, 2.0}).value() == 2.0);
    constexpr auto epi_f = norm.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{1.0, 2.0, 2.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, 1.0}));
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

    // (dakre) TODO refactor
#if 0
    constexpr auto epi_f = affine_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{1.0, 2.0, -1.0}));
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

    // (dakre) TODO refactor
#if 0
    constexpr auto epi_f = affine_fn.epigraph();
    STATIC_REQUIRE(!epi_f.contains(Vectord<4>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{0.0, 0.0, 0.0, 2.0}));
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{1.0, 1.0, 1.0, 16.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<4>{1.0, 1.0, 1.0, 4.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<4>{1.0, 1.0, 0.0, 1.0}));
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

    // (dakre) TODO refactor
#if 0
    constexpr auto hypo_f = affine_fn.hypograph();
    STATIC_REQUIRE(hypo_f.contains(Vectord<5>{}));
    STATIC_REQUIRE(hypo_f.contains(Vectord<5>{0.0, 0.0, 0.0, 0.0, 1.0}));
    STATIC_REQUIRE(!hypo_f.contains(Vectord<5>{1.0, -1.0, -1.0, -1.0, 1.0}));
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
    constexpr auto epi_f = max_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
  }

  SECTION("Test max fn") {
    using Space = Matrixd<1, 5>;
    constexpr auto x = Vectord<5>{-1.0, 4.5, 2.0, 3.0, 5.0};
    constexpr auto max_fn = MaxFn<Space>();
    STATIC_REQUIRE(max_fn.eval(x).has_value());
    STATIC_REQUIRE(max_fn.eval(x).value() == 5.0);
    constexpr auto epi_f = max_fn.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<6>{}));
    STATIC_REQUIRE(
        epi_f.contains(Vectord<6>{x(0), x(1), x(2), x(3), x(4), 5.0}));
    STATIC_REQUIRE(
        !epi_f.contains(Vectord<6>{x(0), x(1), x(2), x(3), x(4), 1.0}));
  }
}

TEST_CASE("Test indicator fn") {
  SECTION("Test indicator fn (hyperplane)") {
    using Space = Matrixd<1, 2>;
    constexpr Vectord<2> pt{1.0, 2.0};
    constexpr double b = 0.5;
    constexpr Hyperplane<Space> hyperplane(pt, b);
    using SetT = linalg::remove_cvref_t<decltype(hyperplane)>;
    constexpr auto Ic = IndicatorFn<SetT>(hyperplane);
    STATIC_REQUIRE(!Ic.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<2>{0.5, 0.0}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<2>{0.5, 0.0}).value() == 0.0);
    // (dakre) TODO refactor
#if 0
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{}, 1.0)));
    STATIC_REQUIRE(epi_f.contains(std::tuple(Vectord<2>{0.5, 0.0}, 1.0)));
    STATIC_REQUIRE(!epi_f.contains(std::tuple(Vectord<2>{0.0, 0.0}, 1.0)));
#endif
  }

  SECTION("Test indicator fn (halfspace)") {
    using Space = Matrixd<1, 2>;
    constexpr Vectord<2> pt{1.0, 2.0};
    constexpr double b = 0.5;
    constexpr Halfspace<Space, HalfspaceType::CLH> halfspace(pt, b);
    using SetT = linalg::remove_cvref_t<decltype(halfspace)>;
    constexpr auto Ic = IndicatorFn<SetT>(halfspace);
    STATIC_REQUIRE(Ic.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<2>{}).value() == 0.0);
    // (dakre) TODO refactor
#if 0
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{0.0, 0.0, 1.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{0.0, 5.0, 1.0}));
#endif
  }

  SECTION("Test indicator fn (lmi)") {
    using Space = Matrixd<1, 3>;
    constexpr Vectord<3> pt{-1.0, 0.0, 1.0};
    constexpr auto I = make_identity<double, 3>();
    constexpr LMI<Space, LMIType::NonStrict> set(I, I, I, I);
    STATIC_REQUIRE(set.is_well_formed());
    using SetT = linalg::remove_cvref_t<decltype(set)>;
    constexpr auto Ic = IndicatorFn<SetT>(set);
    STATIC_REQUIRE(Ic.eval(Vectord<3>{}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<3>{}).value() == 0.0);
    STATIC_REQUIRE(Ic.eval(pt).has_value());
    STATIC_REQUIRE(Ic.eval(pt).value() == 0.0);
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{0.0, 0.0, 0.0, 1.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<4>{0.0, 0.0, 10.0, 1.0}));
  }

  SECTION("Test indicator fn (Rn+)") {
    using Space = Matrixd<1, 1>;
    constexpr NonnegOrthant<Space> rn;
    using SetT = linalg::remove_cvref_t<decltype(rn)>;
    constexpr auto Ic = IndicatorFn<SetT>(rn);
    STATIC_REQUIRE(Ic.eval(Vectord<1>{1.1}).value() == 0.0);
    STATIC_REQUIRE(!Ic.eval(Vectord<1>{-0.1}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<1>{}).value() == 0.0);

    // (dakre) TODO refactor
#if 0
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<2>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<2>{0.0, 1.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<2>{-2.0, 1.0}));
#endif
  }

  SECTION("Test indicator fn (ball)") {
    using Space = Matrixd<1, 2>;
    constexpr auto c = Vectord<2>{0.2, 0.4};
    constexpr double r = 0.6;
    constexpr NormBall<Space, NormBallType::L2CB> ball(c, r);
    using SetT = linalg::remove_cvref_t<decltype(ball)>;
    constexpr auto Ic = IndicatorFn<SetT>(ball);
    STATIC_REQUIRE(Ic.eval(Vectord<2>{}).value() == 0.0);
    STATIC_REQUIRE(!Ic.eval(Vectord<2>{-10.0, 1.0}).has_value());
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{-1.0, -1.0, 1.0}));
  }

  SECTION("Test indicator fn (cone)") {
    using Space = Matrixd<1, 4>;
    constexpr NormCone<Space, NormConeType::L2CC> cone;
    using SetT = linalg::remove_cvref_t<decltype(cone)>;
    constexpr auto Ic = IndicatorFn<SetT>(cone);
    STATIC_REQUIRE(Ic.eval(Vectord<4>{}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<4>{}).value() == 0.0);
    STATIC_REQUIRE(Ic.eval(Vectord<4>{0.1, 0.1, 0.1, 1.0}).value() == 0.0);
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{0.1, 0.1, 0.1, 1.0}));
  }

  SECTION("Test indicator fn (poly)") {
    using Space = Matrixd<1, 2>;
    constexpr auto I = make_identity<double, 2>();
    constexpr auto b = Vectord<2>{1.0, 5.0};
    constexpr Polyhedron<Space, HalfspaceType::CLH, decltype(I)> poly(I, b);
    using SetT = linalg::remove_cvref_t<decltype(poly)>;
    constexpr auto Ic = IndicatorFn<SetT>(poly);
    STATIC_REQUIRE(Ic.eval(Vectord<2>{}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<2>{}).value() == 0.0);
    STATIC_REQUIRE(Ic.eval(Vectord<2>{0.1, 0.2}).value() == 0.0);

    // (dakre) TODO refactor
#if 0
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<3>{0.2, 0.3, 1.0}));
    STATIC_REQUIRE(!epi_f.contains(Vectord<3>{0.2, 0.3, 0.1}));
#endif
  }

  SECTION("Test indicator fn (Sn+)") {
    using Space = SymmetricMatrix<Matrixd<2, 2>>;
    constexpr auto I = make_identity<double, 2>();
    constexpr auto cone = PsdCone<Space>();
    using SetT = linalg::remove_cvref_t<decltype(cone)>;
    constexpr auto Ic = IndicatorFn<SetT>(cone);
    STATIC_REQUIRE(Ic.eval(I).has_value());
    STATIC_REQUIRE(Ic.eval(I).value() == 0.0);
    STATIC_REQUIRE(Ic.eval(Matrixd<2, 2>{}).has_value());
    STATIC_REQUIRE(Ic.eval(Matrixd<2, 2>{}).value() == 0.0);
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Matrixd<2, 2>{}));
    STATIC_REQUIRE(!epi_f.contains(-1.0 * I));
  }

  SECTION("Test indicator fn (simplex)") {
    using Space = Matrixd<1, 3>;
    constexpr auto simplex = Simplex<Space>();
    using SetT = linalg::remove_cvref_t<decltype(simplex)>;
    constexpr auto Ic = IndicatorFn<SetT>(simplex);
    STATIC_REQUIRE(Ic.eval(Vectord<3>{1.0, 0.0, 0.0}).has_value());
    STATIC_REQUIRE(Ic.eval(Vectord<3>{1.0, 0.0, 0.0}).value() == 0.0);
    STATIC_REQUIRE(!Ic.eval(Vectord<3>{0.1, 0.0, 0.0}).has_value());

    // (dakre) TODO refactor
#if 0
    constexpr auto epi_f = Ic.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<4>{0.1, 0.2, 0.7, 1.0}));
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
    constexpr auto epi_f = qol.epigraph();
    STATIC_REQUIRE(epi_f.contains(Vectord<5>{}));
    STATIC_REQUIRE(epi_f.contains(Vectord<5>{1.0, 2.0, 1.0, 2.0, 3.0}));
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
  }
}

TEST_CASE("Test log det fn", "[concave_fn]") {
  SECTION("Test log det fn") {
    constexpr auto I = make_identity<double, 3>();
    constexpr auto log_det = LogDetFn<linalg::SymmetricMatrix<decltype(I)>>();
    STATIC_REQUIRE(log_det(I).has_value());
    STATIC_REQUIRE(log_det(I).value() == 0.0);
    STATIC_REQUIRE(!log_det(-1.0 * I).has_value());
  }
}

TEST_CASE("Test geometric mean fn", "[concave_fn]") {
  SECTION("Test geometric mean fn") {
    using Space = Matrixd<1, 4>;
    constexpr auto geo_mean = GeoMeanFn<Space>();
    STATIC_REQUIRE(geo_mean(Vectord<4>{1.0, 1.0, 1.0, 1.0}) == 1.0);
  }
}
