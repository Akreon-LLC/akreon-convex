# akreon-convex

Header-only C++26 convex modeling and optimization, built on [akreon-linalg](https://github.com/Akreon-LLC/akreon-linalg).

[Documentation](https://akreon-llc.github.io/akreon-convex/) · [Architecture](docs/ARCHITECTURE.md) · [Supported capabilities](docs/ROADMAP.md) · [C++ example](examples/consumer/main.cpp) · [Releases](https://github.com/Akreon-LLC/akreon-convex/releases)

[![DOI](https://zenodo.org/badge/1024451979.svg)](https://doi.org/10.5281/zenodo.23485482)

## Capabilities

Typed variables, DCP expressions, objectives, and constraints are canonicalized into a conic program and solved by a primal-dual predictor-corrector interior-point method. Models and canonicalization support constant evaluation; results include primal/dual iterates, status, numerical events, and optional structural traces.

| Problem class | Support |
|---|---|
| LP, SOCP, SDP | Supported and tested |
| Convex QP and QCQP | Available conic reformulations |
| Exponential and power cones | Partial modeling/canonicalization; solver incomplete |
| Mixed-integer optimization | Planned |

An atom or canonicalization rule alone does not establish solver support. See the [roadmap](docs/ROADMAP.md) for limits.

## Example

The [consumer example](examples/consumer/main.cpp) models and solves a linear program, then checks feasibility and the expected optimum. It also demonstrates installed-package integration.

## Build and test

Requirements:

- CMake 3.30 or newer.
- A C++26-capable compiler with the required constexpr math support.
- Git access for retrieving released dependencies when they are not available locally.

Clone the repository and build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

For local development against an existing Linalg checkout:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg
```

The `host-default` and `host-functional-only` CMake presets are also available.

Unit and functional tests are enabled by default for top-level builds. Disable them with:

```sh
-DBUILD_UNIT_TESTS=OFF -DBUILD_FUNCTIONAL_TESTS=OFF
```

Catch2 3 is used for tests; it is not a production dependency.

## Dependency management

Dependency resolution follows this precedence:

1. Explicit local source checkout using `FETCHCONTENT_SOURCE_DIR_AKREON_LINALG`.
2. Installed `AkreonLinalg` CMake package through `CMAKE_PREFIX_PATH`.
3. CMake FetchContent using the pinned upstream `v0.1.0` release.

Users may override the upstream repository or revision using `AKREON_LINALG_GIT_REPOSITORY` and `AKREON_LINALG_GIT_TAG`.

No Git submodules or prescribed sibling-directory layout are required.

## Use in another CMake project

The public CMake target is `Akreon::Convex`, exported by the `AkreonConvex` package.

```cmake
find_package(AkreonConvex 0.1 CONFIG REQUIRED)

add_executable(my_optimizer main.cpp)
target_link_libraries(my_optimizer PRIVATE Akreon::Convex)
```

Public headers are available under:

```cpp
#include <akreon/convex/convex.hpp>
```

The target propagates its include paths, C++ standard requirement, and `Akreon::Linalg` dependency.

To install the library:

```sh
cmake --install build --prefix /absolute/path/akreon-install
```

Ensure `akreon-linalg` is also installed and discoverable through the same prefix or `CMAKE_PREFIX_PATH`. Installed packages resolve dependencies through CMake without fetching source.

The [consumer example](examples/consumer) demonstrates installed-package discovery and linking.

## Documentation and verification

See [architecture](docs/ARCHITECTURE.md), [support and roadmap](docs/ROADMAP.md), [contributing](CONTRIBUTING.md), and [security](SECURITY.md).

Generate API documentation with Doxygen and Graphviz installed:

```sh
doxygen Doxyfile
```

Open `docs/doxygen/html/index.html`. Tests cover implemented cases, not every dimension or conditioning regime.

## Citation and license

For research citation, use [CITATION.cff](CITATION.cff). Licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.

This repository retains history from the former Akreon SDK. Older revisions carry their own license terms; SDK-era version labels are not standalone Convex releases.
