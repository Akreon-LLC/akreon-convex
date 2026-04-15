# Akreon Cookbook

The is a guide for developing a new optimization problem spec leveraging the core PD-IPM baseline w/ Akreon DCP

## C++ Problem Form
1. First develop a .hpp defining the problem (see control feedback / estimation / etc.)
2. The problem first needs to define / consume a decision variable with an underlying space (i.e., free R^n or PSD, etc.)
3. Next is the problem constraints 1:N which are convex and can leverage atom's and inequalities in the akreon suite
4. After constraints have been added you must decide if the problem is mere feasibility or min. / max. s.t. again convex objective (i.e., c^T z weighted sum or z^T P z + c^T z + q quad form, etc...)
5. Last develop UT to assert problem form is valid and consistent with DCP and is valid (i.e., solvable with PD-IPM)

## Json schema
1. Once the problem is well-defined in c++ you must define the schema in "akreon-schemas" following precedent from prior forms / schemas
2. Add jsonvalidator UT to ensure it is valid

## Python UX
1. In "akreon-python" add the new problem spec s.t. the schema defined above
2. Add UT to ensure problem form is valid and abides by schema above

## Runner Hook
Last is to ensure that there is an appropriate hook into the C++ from outbound UX -> "akreon_runner" to lower to kernel IR
1. Tie in the new schema defined in "akreon_schemas" to "akreon_runner"
2. Develop problem hook to be lowered to kernel IR
3. Depending on problem strucutre allow for build or solve to run in this pipeline
4. Return result (standardized)
5. UT the above + system test(s)
