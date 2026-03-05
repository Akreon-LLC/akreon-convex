#!/usr/bin/env python3
"""Reference LP check in CVXPY test."""

import cvxpy as cp
import numpy as np

# 1. Define the Assets (Market Value)
# House: 500k, Car: 30k, 401k: 200k, Savings: 50k
asset_values = np.array([500000, 30000, 200000, 50000])
n = len(asset_values)

# 2. Define Decision Variables
# x[i] = 1 if Spouse A gets the asset, 0 if Spouse B gets it
x = cp.Variable(n, boolean=True)

# 3. Calculate Totals
total_a = cp.sum(cp.multiply(x, asset_values))
total_b = cp.sum(cp.multiply(1 - x, asset_values))

# 4. Objective: Minimize the absolute difference between totals
# We want total_a to be as close to total_b as possible
objective = cp.Minimize(cp.abs(total_a - total_b))

# 5. Constraints (Optional)
# Example: Spouse A MUST keep the house (asset index 0)
constraints = [x[0] == 1]

# 6. Solve
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GLPK_MI)  # Uses a Mixed-Integer solver

print(f"Spouse A gets assets: {x.value}")
print(f"Spouse A Total: ${total_a.value:,.2f}")
print(f"Spouse B Total: ${total_b.value:,.2f}")
print(f"Difference: ${abs(total_a.value - total_b.value):,.2f}")
