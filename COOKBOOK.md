# Convex problem cookbook

1. Define a typed decision variable and its coordinate space.
2. Compose convex atoms and constraints with valid DCP curvature.
3. Choose feasibility or a convex objective.
4. Canonicalize and solve with the primal-dual interior-point API.
5. Add tests for the model, canonicalization, and numerical result.

Control synthesis models that use this API live in akreon-control. Runtime
bindings, schemas, and deployment belong to their respective downstream projects.
