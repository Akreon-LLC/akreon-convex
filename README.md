# akreon-convex

Optimization domains, DCP atoms and expression composition, convex sets and cones, canonicalization, convex problems, primal-dual interior-point solving, Schur/KKT machinery, and solver diagnostics.

Header-only C++26 library. Public target: `Akreon::Convex`; installed package:
`AkreonConvex`.
Modeling and solver APIs use `convex` and `conic`, with solver settings in `config`.
Public headers live under `include/akreon/convex/`. Only canonical `akreon/...` include paths are supported.

## Compatibility

For supported problem classes, modeling/solver limitations, and planned
extensions, see the [support matrix and roadmap](docs/ROADMAP.md).

## Build and test

Use CMake 3.30+ (for the exported `cxx_std_26` requirement) and a C++26-capable compiler with constexpr math support.
From this checkout:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_SOURCE_DIR_AKREON_LINALG=/absolute/path/akreon-linalg
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

`host-default` and `host-functional-only` presets are also available. Pass the
same local-source overrides when configuring presets. Tests default on only for
the top-level project. Disable tests using `-DBUILD_UNIT_TESTS=OFF
-DBUILD_FUNCTIONAL_TESTS=OFF`.

## Dependencies

The library depends on `akreon-linalg` (`Akreon::Linalg`).
Each dependency resolves in this order:

1. An explicit local checkout via `FETCHCONTENT_SOURCE_DIR_AKREON_LINALG`.
2. An installed CMake package found through `CMAKE_PREFIX_PATH`.
3. FetchContent from the corresponding `Akreon-LLC/akreon-<name>` Git repository,
   pinned to the planned `v0.1.0` release.

The build command above uses local checkouts. No sibling-directory layout or
Git submodules are required. Until release tags exist, use local checkouts or
installed packages. For private Git access or another compatible release, set
`AKREON_LINALG_GIT_REPOSITORY` / `AKREON_LINALG_GIT_TAG`.

Unit tests use installed Catch2 3 or fetch v3.8.1. For offline testing, pass
`-DFETCHCONTENT_SOURCE_DIR_CATCH2=/absolute/path/to/Catch2`.

## Install and consume

The [consumer example](examples/consumer) shows how a separate CMake project
uses the installed library. Ensure `akreon-linalg` is installed
into the same prefix (or include their prefixes in `CMAKE_PREFIX_PATH`). After
building above, install this package, then configure, build, and test the example:

```sh
cmake --install build --prefix /absolute/path/akreon-install
cmake -S examples/consumer -B build/consumer \
  -DCMAKE_PREFIX_PATH=/absolute/path/akreon-install
cmake --build build/consumer
./build/consumer/consumer
ctest --test-dir build/consumer --output-on-failure
```

The example's [CMakeLists.txt](examples/consumer/CMakeLists.txt) finds the package
and links its `consumer` executable:

```cmake
find_package(AkreonConvex 0.1 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Akreon::Convex)
```

Its [main.cpp](examples/consumer/main.cpp) creates and solves a linear program:
minimize `x0 + 2*x1` subject to `x0 >= 1`, `x1 >= 2`, and `x0 + x1 <= 6`.
It checks solver status, feasibility, and the expected optimum `x = [1, 2]`
with objective value `5`, printing the solution and returning nonzero on failure.
Follow the same pattern in your own project, setting `CMAKE_PREFIX_PATH` to the
installation prefix. `Akreon::Linalg` propagates through
`Akreon::Convex`, along with include directories and the C++ standard requirement.
Installed packages resolve their dependencies without fetching source.

## Documentation

The [architecture guide](docs/ARCHITECTURE.md) is the Doxygen landing page.
Contributor workflow is in [CONTRIBUTING.md](CONTRIBUTING.md); test coverage is
in [tests/README.md](tests/README.md).
API and algorithm documentation is generated from the header comments. To build
both together, install Doxygen and Graphviz, then run from this repository:

```sh
doxygen Doxyfile
```

Open `docs/doxygen/html/index.html`. The same generated HTML directory can be
published to GitHub Pages; no separate copy of the architecture guide is needed.

## Citation

If you use this software in research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

[![DOI](https://zenodo.org/badge/1024451979.svg)](https://doi.org/10.5281/zenodo.23485482)

## Project history and licensing

`akreon-convex` evolved from the former Akreon SDK. Its Git history is retained
to preserve development provenance from before the repository split.

Some historical revisions contain the SDK's former proprietary license and
related legal notices. These describe the earlier licensing structure, rather
than the terms for the current `akreon-convex` source.

**The current `main` branch is distributed under the [Apache License 2.0](LICENSE)**,
with attribution in [NOTICE](NOTICE).

When using an older commit, consult the license and notices associated with that
revision. Earlier SDK-era version labels do not represent standalone
`akreon-convex` releases.
