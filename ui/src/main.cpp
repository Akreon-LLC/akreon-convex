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

#if 0
#include <ftxui/component/component.hpp>          // Component, Renderer
#include <ftxui/component/screen_interactive.hpp> // ScreenInteractive
#include <ftxui/dom/elements.hpp>                // text, vbox, border, center

int main() {
  using namespace ftxui;

  // Build a simple component:
  auto component = Renderer([&] {
    return vbox({
             text("Hello from FTXUI 👋"),
             text("This is your minimal boilerplate."),
           }) |
           border |
           center;
  });

  // Create a screen and run the loop:
  auto screen = ScreenInteractive::TerminalOutput();
  screen.Loop(component);

  return 0;
}
#endif

int main() {
  return 0;
}
