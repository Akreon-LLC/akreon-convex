#!/usr/bin/env python3
"""Reference QP check in CVXPY for the 2-var norm-ball test."""

import numpy as np
import cvxpy as cp


def prob1() -> None:
    """prob1
    min. w^T x
    s.t. ||x - c||_2 <= 1
    w = [1, 1]
    b = [3, 1]
    """
    print("***problem 1***")
    c = np.array([3.0, 1.0], dtype=float)
    r = 1.0
    w = np.array([1.0, 1.0], dtype=float)

    x = cp.Variable(2)
    soc = cp.SOC(r, x - c)
    prob = cp.Problem(cp.Minimize(w @ x), [soc])
    obj = prob.solve(verbose=False)

    print(f"status: {prob.status}")
    print(f"solver: {prob.solver_stats.solver_name}")
    print(f"objective: {obj:.16g}")
    print(f"x*: {x.value}")

    primal_violation = np.linalg.norm(x.value - c) - r
    print(f"primal_violation ||x-c||-r: {primal_violation:.3e}")

    # CVXPY SOC dual shape: [lambda_t, lambda_x]
    lam_t = float(np.array(soc.dual_value[0]).reshape(-1)[0])
    lam_x = np.array(soc.dual_value[1]).reshape(-1)
    y_soc = np.hstack([lam_x, lam_t])  # [u1, u2, t] to match Akreon SOC order
    print(f"y_soc [u1,u2,t]: {y_soc}")

    # Conic-form reconstruction for this specific model:
    # F = [[1,0],[0,1],[0,0]], c_obj = [1,1], so rd = F^T y - c_obj
    rd = np.array([y_soc[0] - 1.0, y_soc[1] - 1.0], dtype=float)
    print(f"rd (SOC-conic reconstruction): {rd}")
    print(f"||rd||2: {np.linalg.norm(rd):.3e}")

    # Smooth KKT check for original norm-constraint form
    norm_term = np.linalg.norm(x.value - c)
    grad_norm = (x.value - c) / max(norm_term, 1e-15)
    stationarity = w + lam_t * grad_norm
    complementarity = lam_t * (norm_term - r)
    print(f"lambda: {lam_t:.16g}")
    print(f"stationarity residual: {stationarity}")
    print(f"||stationarity||2: {np.linalg.norm(stationarity):.3e}")
    print(f"complementarity: {complementarity:.3e}")


def prob2() -> None:
    """prob1
    min. c^T x
    s.t. ||x||_2 <= 1
    c = [1, 1]
    """
    print("***problem 2***")
    r = 1.0
    w = np.array([1.0, 1.0], dtype=float)

    x = cp.Variable(2)
    l2 = cp.norm(x, 2) <= r
    prob = cp.Problem(cp.Minimize(w @ x), [l2])
    obj = prob.solve(verbose=False)

    print(f"status: {prob.status}")
    print(f"solver: {prob.solver_stats.solver_name}")
    print(f"objective: {obj:.16g}")
    print(f"x*: {x.value}")


if __name__ == "__main__":
    prob1()
    prob2()
