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

#include "convex/solver/pd_ipm_output.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "math.hpp"

using namespace convex;
using namespace config;
using namespace linalg;

TEST_CASE("PD-IPM output serializes result as JSON", "[pd_ipm_output]") {
  using Var = Vectord<3>;
  using ConeVar = Vectord<2>;
  using NuVar = Vectord<1>;

  PrimalDualResult<Var, ConeVar, NuVar> res{};
  res.tag = SolverResult::OPTIMAL;
  res.event = NumericalEvent::NONE;
  res.stalled = false;
  res.num_iter = 7;
  res.mu = 1e-6;
  res.convex_obj = 1.25;
  res.dual_obj = 1.24;
  res.iterate.z = Var{1.0, 2.0, 3.0};
  res.iterate.x = ConeVar{4.0, 5.0};
  res.iterate.y = ConeVar{6.0, 7.0};
  res.iterate.nu = NuVar{8.0};
  res.rd = Var{0.1, 0.2, 0.3};
  res.rp = ConeVar{0.4, 0.5};
  res.req = NuVar{0.6};

  char buf[8192];
  const auto len = result_to_json_cstr(res, buf, sizeof(buf));
  REQUIRE(len > 0);

  const std::string json(buf);
  REQUIRE(json.starts_with("{"));
  REQUIRE(json.find("\"tag\":\"OPTIMAL\"") != std::string::npos);
  REQUIRE(json.find("\"tag_code\":1") != std::string::npos);
  REQUIRE(json.find("\"event\":\"NONE\"") != std::string::npos);
  REQUIRE(json.find("\"stalled\":false") != std::string::npos);
  REQUIRE(json.find("\"num_iter\":7") != std::string::npos);
  REQUIRE(json.find("\"convex_obj\":1.25") != std::string::npos);
  REQUIRE(json.find("\"dual_obj\":1.24") != std::string::npos);
  REQUIRE(json.find("\"iterate\":{\"z\":[1, 2, 3],\"x\":[4, 5],\"y\":[6, "
                    "7],\"nu\":[8]}") != std::string::npos);
  REQUIRE(json.find("\"residuals\":{\"rd\":[0.1, 0.2, 0.3],\"rp\":[0.4, "
                    "0.5],\"req\":[0.6]}") != std::string::npos);
}

TEST_CASE("PD-IPM output serializes synthesized gain object as JSON",
          "[pd_ipm_output]") {
  using Var = Vectord<2>;
  using ConeVar = Vectord<1>;
  using NuVar = Vectord<0>;
  using Gain = Matrixd<2, 2>;

  PrimalDualResult<Var, ConeVar, NuVar> res{};
  res.tag = SolverResult::OPTIMAL;
  res.event = NumericalEvent::NONE;
  res.iterate.z = Var{1.0, 2.0};
  res.iterate.x = ConeVar{3.0};
  res.iterate.y = ConeVar{4.0};
  constexpr auto K = Gain{Vectord<2>{-1.0, 0.5}, Vectord<2>{0.25, -2.0}};

  char buf[8192];
  const auto len =
      result_to_json_cstr(res, "state_feedback_gain", K, buf, sizeof(buf));
  REQUIRE(len > 0);

  const std::string json(buf);
  REQUIRE(json.find("\"tag\":\"OPTIMAL\"") != std::string::npos);
  REQUIRE(json.find("\"synthesized_object\":{\"kind\":\"state_feedback_gain\","
                    "\"matrix\":[[-1, 0.5],[0.25, -2]]}") !=
          std::string::npos);
}

TEST_CASE("PD-IPM output reports JSON to stream and file", "[pd_ipm_output]") {
  using Var = Vectord<2>;
  using ConeVar = Vectord<1>;
  using NuVar = Vectord<0>;

  PrimalDualResult<Var, ConeVar, NuVar> res{};
  res.tag = SolverResult::PRIMAL_FEASIBLE;
  res.event = NumericalEvent::NUMERICAL_SAFEGUARD;
  res.stalled = true;
  res.num_iter = 3;
  res.iterate.z = Var{9.0, 10.0};
  res.iterate.x = ConeVar{11.0};
  res.iterate.y = ConeVar{12.0};
  res.rd = Var{0.01, 0.02};
  res.rp = ConeVar{0.03};

  std::ostringstream out;
  report_result_json(out, res);
  const auto stream_json = out.str();
  REQUIRE(stream_json.find("\"tag\":\"PRIMAL_FEASIBLE\"") != std::string::npos);
  REQUIRE(stream_json.find("\"event\":\"NUMERICAL_SAFEGUARD\"") !=
          std::string::npos);

  const auto path =
      std::filesystem::temp_directory_path() / "akreon_pd_ipm_output_test.json";
  REQUIRE(write_result_json(path, res));

  std::ifstream in(path);
  REQUIRE(in.is_open());
  const std::string file_json((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
  REQUIRE(file_json == stream_json);

  std::filesystem::remove(path);
}

TEST_CASE("PD-IPM output reports synthesized gain object to stream and file",
          "[pd_ipm_output]") {
  using Var = Vectord<1>;
  using ConeVar = Vectord<1>;
  using NuVar = Vectord<0>;
  using Gain = Matrixd<1, 2>;

  PrimalDualResult<Var, ConeVar, NuVar> res{};
  res.tag = SolverResult::DUAL_FEASIBLE;
  res.event = NumericalEvent::NONE;
  res.iterate.z = Var{1.0};
  res.iterate.x = ConeVar{2.0};
  res.iterate.y = ConeVar{3.0};
  constexpr auto L = Gain{Vectord<2>{4.5, -0.75}};

  std::ostringstream out;
  report_result_json(out, res, "observer_gain", L);
  const auto stream_json = out.str();
  REQUIRE(stream_json.find(
              "\"synthesized_object\":{\"kind\":\"observer_gain\","
              "\"matrix\":[[4.5, -0.75]]}") != std::string::npos);

  const auto path = std::filesystem::temp_directory_path() /
                    "akreon_pd_ipm_output_gain_test.json";
  REQUIRE(write_result_json(path, res, "observer_gain", L));

  std::ifstream in(path);
  REQUIRE(in.is_open());
  const std::string file_json((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
  REQUIRE(file_json == stream_json);

  std::filesystem::remove(path);
}
