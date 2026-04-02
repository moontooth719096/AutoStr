---
name: autostr-cli-workflow
description: "Use when editing AutoStr command-line options, help text, defaults, or argument parsing in main.py."
---

# AutoStr CLI Workflow

Use this skill when a task touches the command-line interface.

## Goal

Keep the parser, help text, README examples, and CLI tests in sync.

## Workflow

1. Read `main.py` first to see the current parser and defaults.
2. Check `README.md` for the user-facing examples and option descriptions.
3. Check `tests/test_cli.py` for the expected argument behavior.
4. Make the smallest parser change that matches the requested behavior.
5. Verify that the new flag, default, or help text stays consistent across code, docs, and tests.

## Typical Files To Check

- `main.py`
- `README.md`
- `tests/test_cli.py`

## When To Use

- A user asks to add, rename, or remove a CLI flag.
- A default value or help message changes.
- The command-line behavior needs to stay aligned with the README.

## What Not To Do

- Do not change a flag without checking existing test coverage.
- Do not update the parser and forget the documentation.
- Do not assume a CLI option is safe to rename if other code depends on it.