/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#include "model/state_space.hpp"

namespace control {
/**
 * @brief LQR
 *
 * Performance of the system V(x,t) = min_u ∫ [x(t)^T Q x(t) + u(t)^T R u(t)] dt
 * Q \in S^n_+ penalizes state deviations
 * R \in S^n_+ penalizes control effort
 *
 * V(x) = x^T P x
 * 0 = min_u[ x^T Q x + u^T R u + \nabla V^T(A x + B u) ]
 * \nabla V = 2 P x
 * so we have... u^T R u + 2 x^T P B u
 * derivative w.r.t u and set to zero... u^* = -R^{-1} B^T P x
 *
 * Goal find u(t) that minimizes J via Ricatti EQ
 * u^*(t) = -K x(t)... K = R^{-1} B^T P
 * where K is the feedback gain controller
 *
 * sub back into HJB: 0 = x^T(Q + A^T P + P A - P B R^{-1} B^T P) x
 *
 * Therefore CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
 */
template <ContinuousStateSpaceLike SS>
struct LQR {};

/**
 * @brief DLQR
 */
template <DiscreteStateSpaceLike SS>
struct DLQR {};

}  // namespace control
