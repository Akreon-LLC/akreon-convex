#include <akreon/convex/convex.hpp>

#include <cmath>
#include <iostream>

int main() {
  // Minimize x0 + 2*x1 subject to x0 >= 1, x1 >= 2, x0 + x1 <= 6.
  // The unique optimum is x = [1, 2], with objective value 5.
  using Space = linalg::Matrix<double, 1, 2>;
  constexpr auto x = convex::Variable<convex::VarMeta<Space>>();
  constexpr auto A =
      linalg::Matrix<double, 3, 2>{linalg::Vector<double, 2>{-1.0, 0.0},
                                   linalg::Vector<double, 2>{0.0, -1.0},
                                   linalg::Vector<double, 2>{1.0, 1.0}};
  constexpr auto b = linalg::Vector<double, 3>{-1.0, -2.0, 6.0};
  constexpr auto constraints = convex::Polyhedron(A, b, x);
  constexpr auto cost =
      convex::WeightedSum<Space>()(linalg::Vector<double, 2>{1.0, 2.0}, x);
  constexpr auto problem =
      convex::make_problem(x).in_set(constraints).minimize(cost);

  // The model above is canonicalized at compile time; solve it at runtime.
  const convex::PrimalDualIPM solver(problem);
  const auto& result = solver.res;
  if (result.tag != config::SolverResult::OPTIMAL ||
      linalg::is_fatal_event(result.event) || result.stalled) {
    std::cerr << "Optimization failed: status=" << static_cast<int>(result.tag)
              << ", event=" << static_cast<int>(result.event) << '\n';
    return 1;
  }

  const auto x0 = result.iterate.z(0);
  const auto x1 = result.iterate.z(1);
  const auto objective = x0 + 2.0 * x1;
  constexpr double tolerance = 1e-3;
  if (!std::isfinite(x0) || !std::isfinite(x1) ||
      !std::isfinite(result.convex_obj) || std::abs(x0 - 1.0) > tolerance ||
      std::abs(x1 - 2.0) > tolerance || x0 < 1.0 - tolerance ||
      x1 < 2.0 - tolerance || x0 + x1 > 6.0 + tolerance ||
      std::abs(objective - 5.0) > tolerance ||
      std::abs(result.convex_obj - objective) > tolerance) {
    std::cerr << "Unexpected solution: x=[" << x0 << ", " << x1
              << "], objective=" << objective
              << ", reported objective=" << result.convex_obj << '\n';
    return 1;
  }

  std::cout << "Optimal solution: x=[" << x0 << ", " << x1
            << "], objective=" << objective << '\n';
  return 0;
}
