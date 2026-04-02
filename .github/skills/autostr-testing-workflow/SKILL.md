---
name: autostr-testing-workflow
description: "Use when changing AutoStr behavior and you need to verify it with the most relevant tests first; confirm the active Python environment, then run focused tests with a single pytest entrypoint."
---

# AutoStr Testing Workflow

Use this skill when a change needs proof that the current behavior still works.

## Goal

Pick the smallest useful test set, run it with a known Python interpreter, and expand only if the result is unclear.

## Workflow

1. Identify the module or feature that changed.
2. Confirm the active Python interpreter before testing, and use that same interpreter for the whole run.
3. Prefer `python -m pytest` as the default entrypoint so the tests run in the same environment as the selected Python.
4. Read the matching tests before changing the code.
5. Prefer focused pytest targets over running the whole suite immediately.
6. If terminal output is incomplete, rely on the exit code and a captured log rather than guessing from partial output.
7. If a failure appears, trace it back to the smallest relevant function or branch.
8. After the fix, rerun the same tests to confirm the result.

## Typical Files To Check

- `tests/test_cli.py`
- `tests/test_pipeline_batch.py`
- `tests/test_pipeline_highlights.py`
- `tests/test_highlight.py`
- `tests/test_reflow_and_srt.py`
- `autostr/pipeline.py`
- `autostr/highlight.py`
- `autostr/reflow.py`

## When To Use

- You changed a parser option, pipeline step, or output format.
- You want to confirm a bug fix without waiting on unrelated tests.
- You need to know which test file best represents the current behavior.

## What Not To Do

- Do not switch between `pytest`, `py -m pytest`, and `python -m pytest` within the same investigation unless you are debugging the environment itself.
- Do not start with the whole test suite unless the change is broad.
- Do not rely on memory when a matching test already exists.
- Do not treat a passing unrelated test as proof that the changed path works.