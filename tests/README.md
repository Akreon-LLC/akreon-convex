# Verification

Use the [README build/test commands](../README.md#build-and-test) for configuration
and execution. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for formatting and CI.

## Suite coverage

- Unit tests cover atoms, constraints, canonicalization, cone operations, and
  LP/QP/SOCP/SDP solves. Quadratic constraints are exercised in `unit/qp_test.cpp`.
- Functional tests cover randomized LP/QP/SDP solves and quadratic forms.
- Atom evaluation/canonicalization tests alone do not establish solver support;
  feature status belongs in the roadmap.
- `Akreon::ConvexTestSupport` supplies result logging, including traces attached
  to the result; generic helpers come from `Akreon::LinalgTestSupport`.

The [consumer example](../examples/consumer) verifies installed-package consumption;
its configure/build/run commands are maintained in the README. Numerical coverage
describes regression tests, not proof for all dimensions or input conditioning.
See the [support roadmap](../docs/ROADMAP.md) for compatibility and known gaps.
