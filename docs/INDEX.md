# Akreon Convex

Typed convex modeling, conic canonicalization, and primal-dual optimization in C++26.

## Start here

- [Build, install, and examples](https://github.com/Akreon-LLC/akreon-convex#readme)
- [Architecture](ARCHITECTURE.md): source organization and design.
- [Support and roadmap](ROADMAP.md): implemented capabilities and limitations.
- \ref convex "API reference": types and functions.

## Workflow

| Area | Purpose |
|---|---|
| Model | Compose fixed-dimension variables, DCP expressions, sets, objectives, and constraints. |
| Canonicalize | Convert supported compositions into affine constraints and product cones. |
| Solve | Inspect solver status, primal/dual iterates, numerical events, and diagnostics. |

## C++ integration

```cpp
#include <akreon/convex/convex.hpp>
```

Link the `Akreon::Convex` CMake target. The library is header-only; dimensions and
mathematical contracts are represented in C++ types.

LP, SOCP, and SDP are supported; QP/QCQP support depends on available conic reformulations. Exponential/power cone solving is incomplete.
