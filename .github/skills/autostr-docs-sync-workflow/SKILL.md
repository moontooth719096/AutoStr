---
name: autostr-docs-sync-workflow
description: "Use when updating AutoStr README examples, usage notes, or option descriptions so the docs stay in sync with the code."
---

# AutoStr Docs Sync Workflow

Use this skill when code and documentation should change together.

## Goal

Keep the README, CLI help text, and tests consistent with the current implementation.

## Workflow

1. Read the user-facing text in `README.md` and the parser in `main.py`.
2. Check whether a test already captures the behavior being documented.
3. Update the code and the documentation together when the public behavior changes.
4. Prefer short examples that reflect the current defaults and file paths.
5. Remove or revise examples that no longer match the current repo.

## Typical Files To Check

- `README.md`
- `main.py`
- `tests/test_cli.py`
- `tests/test_pipeline_highlights.py`
- `tests/test_pipeline_batch.py`

## When To Use

- A flag, default, or workflow description changes.
- An example command in the README no longer matches the parser.
- You want the docs to explain the current behavior, not an old version of it.

## What Not To Do

- Do not update examples without checking the actual argument names.
- Do not leave stale instructions in the README after a behavior change.
- Do not document a feature that has not been verified in the workspace.