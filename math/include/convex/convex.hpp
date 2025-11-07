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

#pragma once

#include "cone_view.hpp"
#include "convex_ops.hpp"
#include "dual_cone_view.hpp"
#include "functions/affine_fn.hpp"
#include "functions/exp_fn.hpp"
#include "functions/geo_mean_fn.hpp"
#include "functions/indicator_fn.hpp"
#include "functions/log_det_fn.hpp"
#include "functions/log_sum_exp_fn.hpp"
#include "functions/max_fn.hpp"
#include "functions/norm_fn.hpp"
#include "functions/quad_over_linear_fn.hpp"
#include "sets/affine_subspace.hpp"
#include "sets/ellipsoid.hpp"
#include "sets/exp_cone.hpp"
#include "sets/halfspace.hpp"
#include "sets/hyperplane.hpp"
#include "sets/lmi.hpp"
#include "sets/log_det_cone.hpp"
#include "sets/log_sum_exp_cone.hpp"
#include "sets/neg_orthant.hpp"
#include "sets/nonneg_orthant.hpp"
#include "sets/norm_ball.hpp"
#include "sets/norm_cone.hpp"
#include "sets/polyhedron.hpp"
#include "sets/power_cone.hpp"
#include "sets/psd_cone.hpp"
#include "sets/rsoc.hpp"
#include "sets/simplex.hpp"
#if 0
#include "functions/log_fn.hpp"

// TODO - need slacks for these
#include "cone_program.hpp"
#include "convex_program.hpp"
#endif
