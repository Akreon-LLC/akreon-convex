# Four-repository migration

The existing convex Git repository and history are retained in place. The three
other existing repository directories receive source moves, not copied algorithm
implementations. No commits, tags, remote changes, history rewriting, or Git
registration changes are part of this working-tree migration.

| Original ownership in the monolith | New owner |
| --- | --- |
| `core/include/linalg/**` (including affine maps and matrix exponential) | akreon-linalg `include/akreon/linalg/**` |
| Assertions, `config::FactorConfig`, `linalg::NumericalEvent` | akreon-linalg `assert.hpp`, `numerical_config.hpp` |
| `core/include/convex/**`, remaining solver configuration | akreon-convex `include/akreon/convex/**` |
| `core/include/control/**`, `control.hpp` | akreon-control `include/akreon/control/**` |
| `core/include/dynamics/**`, `dynamics.hpp` | akreon-dynamics `include/akreon/dynamics/**` |
| Linalg taxonomy and API reference | akreon-linalg `docs/CODE_REFERENCE.md` |
| Convex canonicalization and solver notes | akreon-convex `docs/` |
| Control synthesis and gain recovery notes | akreon-control `docs/` |

Each public header has one canonical path under `include/akreon/<domain>/`.
The old unprefixed include trees and combined `math.hpp` umbrella are removed.
Consumers include the domain headers they use explicitly. Namespaces and
numerical algorithms are preserved.

## Dependency contracts

```text
Akreon::Linalg -> Akreon::Convex -> Akreon::Control
Akreon::Linalg -----------------> Akreon::Control
Akreon::Linalg -----------------> Akreon::Dynamics
```

The arrows run from dependency to consumer. There are no reverse edges and no
control or convex dependency in dynamics. Each package is a C++26 INTERFACE
library. CMake 3.30 is required to export `cxx_std_26`.

| Repository | Package | Target |
| --- | --- | --- |
| akreon-linalg | AkreonLinalg | Akreon::Linalg |
| akreon-convex | AkreonConvex | Akreon::Convex |
| akreon-control | AkreonControl | Akreon::Control |
| akreon-dynamics | AkreonDynamics | Akreon::Dynamics |

Generic test support is owned by linalg (`Akreon::LinalgTestSupport`). Convex
owns the solver logging helper (`Akreon::ConvexTestSupport`). Their headers are
installed under `share/akreon-*/test-support`, outside production include roots.
This lets each dependent repository run its own suite against installed packages
without duplicating support code or adding production dependency edges.

## Verification ownership

- Linalg: affine, linalg, linalg contracts, linear systems (matrix exponential
  and generic solves), and decomposition fuzz tests.
- Convex: convex functions, problems and sets, cones, LP/QP/SOCP/SDP, four
  functional tests, and the canonicalization CTAD lint.
- Control: controllers, plant models, state feedback and observers, including
  polytopic variants.
- Dynamics: one- and two-DoF pendulums and the spring-mass-damper model.

The PSD Jordan-trace case formerly embedded in `linalg_test.cpp` now lives in
`cone_test.cpp`. Splitting the old umbrella exposed missing direct includes in
`variable.hpp`, `canonical_form_builder.hpp`, and `sets/lmi.hpp`; these now
include their actual convex dependencies.

No standalone examples or benchmarks were present in the original checkout.
Each repository now includes `examples/consumer`, which imports only its
installed package and compiles its public umbrella. CI runs those consumers
after installing the package and its dependencies.

## Development and installation

From a workspace containing independent checkouts:

```sh
cmake -S akreon-control -B build/control \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_CONVEX=/absolute/path/akreon-convex
cmake --build build/control --parallel 2
ctest --test-dir build/control --output-on-failure

cmake -S akreon-convex -B build/convex \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg
```

Use the same linalg override for dynamics. Linalg needs no Akreon dependency.
To install without tests, configure each repository with `BUILD_UNIT_TESTS=OFF`
and `BUILD_FUNCTIONAL_TESTS=OFF`. Install in dependency order to a chosen prefix:

```sh
cmake --install build/linalg --prefix /absolute/path/install
cmake --install build/convex --prefix /absolute/path/install
cmake --install build/control --prefix /absolute/path/install
cmake --install build/dynamics --prefix /absolute/path/install
cmake -S akreon-control/examples/consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH=/absolute/path/install
cmake --build build/consumer
ctest --test-dir build/consumer --output-on-failure
```

Installed CONFIG packages use `find_dependency` for dependencies not already
provided as targets. Local checkouts, installed packages, and mixed resolution
are supported. Package consumers do not fetch dependencies implicitly.

## Validation and release follow-up

Local validation uses AppleClang 21, CMake 4.4, Catch2 3.8.1, and clang-format 22:
27 independent CTest tests plus four installed consumer tests pass. Resolution
checks cover local override priority over deliberately failing installed
packages, installed packages with unusable Git URLs, and mixed local linalg /
installed convex. CI retains GCC and Clang verification; those toolchains have
not been run locally during this migration.

The `v0.1.0` fallback is a planned release pin. Remote fetching cannot work until
compatible repository contents and tags are published to the configured remotes
and the caller has access. Public visibility is not required: local overrides,
installed packages, and authenticated private Git are supported now. Do not
claim the remote fallback has been exercised against a published release.
CI dependency checkouts need a read token while private; release refs can replace
the documented development refs once available. No new numerical algorithms,
new optimization/control features, or additional repository edges were added.
