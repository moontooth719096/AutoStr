---
name: autostr-adaptive-workflow
description: "Use when working on AutoStr tasks that may have changed since the last edit; re-read the current project files first, then validate changes against tests and the latest code."
---

# AutoStr Adaptive Workflow

Use this skill for changes in a moving codebase where the latest files matter more than any cached assumption.

## Goal

Keep edits aligned with the current repository state, even when the project has changed since the last task, by rereading the latest code before making changes.

## Workflow

1. Re-read the files that define the current behavior before making assumptions.
2. Re-read the most relevant nearby tests before editing, so the current expectations are fresh.
3. Prefer source code and tests over memory, comments, or stale examples.
4. If documentation disagrees with code, treat the code as the source of truth and report the mismatch.
5. After editing, run or review the most relevant tests for the changed area.
6. Summarize any behavior changes clearly so the user can verify them quickly.

## Typical Files To Check

- `README.md`
- `main.py`
- `autostr/pipeline.py`
- `autostr/transcribe.py`
- `tests/`

## When To Use

- The user asks for a fix, but the repo may have changed since the last explanation.
- The task depends on current CLI flags, pipeline behavior, or test expectations.
- You want a repeatable workflow that stays valid as the project evolves.

## What Not To Do

- Do not hard-code outdated assumptions into the skill.
- Do not skip re-reading the current files.
- Do not carry over old behavior from a previous task without confirming the current code.
- Do not describe an implementation as current unless it was verified in the workspace.