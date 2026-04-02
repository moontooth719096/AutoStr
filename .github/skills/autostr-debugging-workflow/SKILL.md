---
name: autostr-debugging-workflow
description: "Use when AutoStr has a failing test, wrong output, or runtime bug and you need to find the root cause first."
---

# AutoStr Debugging Workflow

Use this skill when something is broken and the first job is to understand why.

## Goal

Reproduce the problem, isolate the code path, and fix the smallest true cause.

## Workflow

1. Reproduce the bug with the smallest reliable command or test.
2. Read the code path that handles the failing case.
3. Compare the code against nearby tests and expected output.
4. Fix the root cause instead of applying a broad workaround.
5. Re-run the focused test or command to confirm the bug is gone.

## Typical Files To Check

- `main.py`
- `autostr/pipeline.py`
- `autostr/highlight.py`
- `autostr/reflow.py`
- `autostr/transcribe.py`
- `tests/`

## When To Use

- A test fails and the failure message points to one area.
- Subtitle timing, highlight export, or batch scanning behaves incorrectly.
- You need to find the reason before deciding how to patch it.

## What Not To Do

- Do not guess at the fix before reproducing the issue.
- Do not rewrite unrelated code while chasing one failure.
- Do not claim a bug is fixed until the failing path has been checked again.