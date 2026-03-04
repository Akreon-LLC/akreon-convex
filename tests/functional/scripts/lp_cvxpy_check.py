#!/usr/bin/env python3
"""Reference LP check in CVXPY test."""

import numpy as np
import cvxpy as cp

# min. c^T x
# s.t. A x <= b
# A = [1 1; 2 1; -1 2]
# b = [4 5 2]


def main() -> None:
    c = np.array([2.0, -1.0], dtype=float)
    b = np.array([4.0, 5.0, 2.0], dtype=float)
    A = np.array([[1.0, 1.0], [2.0, 1.0], [-1.0, 2.0]], dtype=float)

    x = cp.Variable(2)
    poly = A @ x <= b
    obj = cp.Minimize(c @ x)
    prob = cp.Problem(obj, [poly, x >= 0])
    res = prob.solve(verbose=False)

    print(f"status: {prob.status}")
    print(f"solver: {prob.solver_stats.solver_name}")
    print(f"objective: {res:.16g}")
    print(f"x*: {x.value}")


if __name__ == "__main__":
    main()
