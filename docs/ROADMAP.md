# Support and Roadmap {#convex_support_roadmap}

For expression and solver data flow, see the [architecture guide](ARCHITECTURE.md).

Status distinguishes modeling infrastructure from tested end-to-end solves.
In-progress code remains on main; roadmap items have no delivery commitment.

## Current support

| Problem family | Status |
| --- | --- |
| LP | Supported and tested |
| Convex QP | Supported through available conic reformulations |
| Convex QCQP | Supported for available SOC/rotated-SOC formulations, including ellipsoidal and squared-inequality constraints |
| SOCP, SDP | Supported and tested |
| Exponential/power cones | Partial modeling and canonicalization; asymmetric-cone solver support in progress |
| Mixed-integer programming (MIP) | Planned; integer/binary modeling and solving are not implemented |

QCQP means quadratically constrained quadratic programming. Support is limited
to convex formulations expressible through the existing APIs, not arbitrary
nonconvex quadratics.

## Atom support boundary {#convex_atom_support}

| Atoms | Required cone | Solve status |
| --- | --- | --- |
| `Exp`, `Log`, `Softplus`, `LogSumExp`, `KLDivergence` | Exponential | In progress |
| `GeoMean` | Power | In progress |

Evaluation/canonicalization tests do not establish solver support. Both asymmetric
cones still have placeholder step-length routines and incomplete solver operations.
`GeoMean` currently emits a power cone, not an alternative SOC reformulation.

## Remaining work

- **QCQP:** expand tests combining convex quadratic objectives with multiple
  quadratic constraints, checking canonicalization, feasibility, and optima.
- **Exponential/power cones:** complete cone operations and a compatible
  asymmetric-cone algorithm; audit atom formulations and add end-to-end solves.
- **Mixed integer:** define integer/binary domains, discrete search, bounds/gap
  reporting, and verification against known integer optima.
- **Capability diagnostics:** reject unsupported cone/domain combinations early
  with actionable errors. Current asymmetric solves may fail in template instantiation.

Promote support only after modeling, canonicalization, solver behavior, and result
validation are tested together. Keep API status notes linked to this page.
