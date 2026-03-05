#!/usr/bin/env python3
"""Reference SOCP check in CVXPY for the 2-var norm-ball test."""

import numpy as np
import cvxpy as cp


def prob() -> None:
    """prob
    min. 0.5 ||x + a||_2^2
    s.t. ||x||_2 <= 1
    b = [-1, 2]
    """
    print("***problem 1***")
    c = np.array([-1.0, 2.0], dtype=float)
    r = 1.0

    x = cp.Variable(2)
    soc = cp.SOC(r, x)
    prob = cp.Problem(cp.Minimize(0.5 * cp.square(cp.norm(x + c, 2))), [soc])
    obj = prob.solve(verbose=False)

    print(f"status: {prob.status}")
    print(f"solver: {prob.solver_stats.solver_name}")
    print(f"objective: {obj:.16g}")
    print(f"x*: {x.value}")


if __name__ == "__main__":
    prob()
