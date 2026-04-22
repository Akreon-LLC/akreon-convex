# Agents

## Purpose

`akreon-sdk` is the developer-facing technical SDK for Akreon systems.

This repo includes substantial math, optimization, controls, estimation,
modeling, and analysis to produce automated client-facing code via `.so`
artifacts with C API wrappers and/or solver results.

Treat this repo as a **technical library**, not just an integration layer, while preserving boundaries with `akreon-schemas`, `akreon-runner`, and closed execution systems.

## Principles

- Prefer clarity over cleverness.
- Prefer technical rigor over convenience.
- Preserve stable developer APIs where practical.
- Keep interfaces typed, explicit, and testable.
- Keep numerical and control semantics clear and documented.

## Rules

- Domain logic that improves the developer-facing SDK is appropriate here.
- Do not move closed runner/build/deployment logic into this repo.
- Do not duplicate schema ownership that belongs in `akreon-schemas`.
- Prefer explicit models, configs, and result objects over loose dictionaries.
- Validate inputs early and surface actionable errors.
- Do not introduce hidden global state or surprising defaults.
- Prioritize constexpr functionality always for real-time safety code.

## Style guide

Please abide by the checked-in
[clang-format](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
configuration in this repo.

After changing C++ headers or sources, run:
`clang-format -i <touched-files>`

To audit formatting across the tracked SDK sources, use:
`cmake --build <build-dir> --target clang_format_check`

After code changes, run the relevant build and test flow. For host work, prefer:
`cmake --preset host-default`
`cmake --build --preset host-default -j`
`ctest --test-dir out/build/host-default/tests --output-on-failure`

## When changing code

Before making a change, check:

1. Does this belong in the SDK?
2. Is this clearer for developers?
3. Will this break existing code or workflows?
4. Are assumptions, units, timing, and semantics explicit?
5. Is the behavior tested?
6. Are there regressions across clang, gcc, or target CPU architectures?

## Testing

Changes should include relevant automated coverage in this repo via `ctest`.

## Report

Include:

- Summary
- Changeset
- SDK impact
- Testing
- Gaps
- Security issues and/or leaks, when present
- Ablation study results or regressions, when that analysis was actually run

## Goal

Keep `akreon-sdk` clear, stable, technically rigorous, and easy for developers to use.
