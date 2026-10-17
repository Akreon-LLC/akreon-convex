# Core tests

Run the host suite from the repository root:

```sh
cmake --preset host-default
cmake --build --preset host-default -j
ctest --test-dir out/build/host-default/tests --output-on-failure
```

`unit/` contains Catch2 tests registered with CTest by executable. To inspect
individual cases, run an executable with `--list-tests`; pass a quoted case name
to run just that case.

- `linalg_test` checks numerical operations; `linalg_contract_test` checks type,
  shape, and coordinate contracts; `linear_systems_test` checks decompositions
  and solves.
- `convex_fn_test` and `convex_set_test` check atom evaluation, membership, and
  canonicalization. `convex_prob_test` checks composition and problem assembly.
- `cone_test` checks cone operations and scaling identities. The LP, QP, SOCP,
  and SDP executables check canonical models and solver results together.
- Controller, observer, feedback, and plant tests check control interfaces and
  synthesis. The DOF model tests check dynamics at rest and trajectory regressions.
- `linalg_fuzz_test` runs deterministic float/double QR, eigen decomposition,
  SVD reconstruction, eigenpair residuals, and least-squares stress coverage from `functional/` in CTest, including
  builds with only `BUILD_UNIT_TESTS` enabled. It defaults to 100 trials per
  scalar/shape (2,200 matrices); pass a positive trial count to the executable
  for a longer run, e.g. `linalg_fuzz_test 10000`. Diagnostics include the seed,
  scalar, shape, trial, and failed check, capped at three matrices per shape.
- `functional/` contains the numerical stress and fuzz programs. See
  `CMakeLists.txt` for which programs are registered with CTest.

Name cases after the operation and observable behavior. Use sections for distinct
inputs or scenarios sharing a fixture; a sole section repeating its case name is
unnecessary. Keep compile-time assertions for constexpr contracts and numerical
results that must remain usable at compile time.

Equivalent numerical examples can cover distinct APIs: an L2 norm ball, an
ellipsoid, and a lifted norm inequality exercise different canonicalization
paths. Preserve those cases, as well as separate atom, assembly, and solver
checks. Consolidate assertions only when they exercise the same behavior through
the same path.
