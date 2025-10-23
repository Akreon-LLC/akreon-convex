/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

namespace unit_test {

constexpr double eps = 1e-8;

template <typename T>
concept DefaultConstructible = requires { T{}; };

}  // namespace unit_test
