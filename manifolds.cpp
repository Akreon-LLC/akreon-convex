#include "manifolds.hpp"

#include <assert.h>

using namespace topology;

int main() {
  using T = int;
  Topology<T> tau;

  // Case A
  // Topology on {1,2} with open sets: null, {1}, {2}, {1,2}
  tau.open_sets = {{}, {1}, {2}, {1, 2}};

  Set<T> X = {1, 2};

  static_assert(TopologicalSpace<T>, "This toplogy is invalid!");

  // Case B
  X = {1, 3};
  tau.open_sets = {{}, {1}, {1, 2}, {1, 2, 3}};
  auto int_X = tau.interior(X);
  assert(int_X.size() == 1);

  // Case C
  auto ext_X = tau.exterior(X);
  assert(ext_X.size() == 0);

  // Case D
  auto cl_X = tau.closure(X);
  assert(cl_X.size() == 3);

  // Case E
  auto bd_X = tau.boundary(X);
  assert(bd_X.size() == 2);

  // Case F - Manifold test
  static_assert(!TopologicalManifold<T>, "This manifold is valid!");
  static_assert(TopologicalManifold<double>, "This manifold is invalid!");

  // Case G - Chart test
  using Td = double;               // manifold points
  using Rn = std::vector<double>;  // points \in E

  // Forward map: double -> vector<double>
  auto phi = [](Td x) -> Rn { return {x}; };

  // Inverse map: vector<double> -> double
  auto inv_phi = [](const Rn& y) -> Td { return y[0]; };

  Chart<Td, Rn, decltype(phi), decltype(inv_phi)> chart{
      .U = {1.f, 2.f}, .phi = phi, .inv_phi = inv_phi};

  static_assert(ValidChart<Td, Rn, decltype(phi), decltype(inv_phi)>,
                "This chart is invalid!");

  // Case H - Atlas test
  static_assert(ValidAtlas<Td, Rn, decltype(phi), decltype(inv_phi)>,
                "This atlas is invalid");

  return 0;
}
