/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

namespace unit_test {

constexpr double eps = 1e-10;

template <typename T>
concept DefaultConstructible = requires { T{}; };

}  // namespace unit_test
