---
name: autostr-docker-workflow
description: "Use when editing AutoStr Dockerfile, compose files, runtime environment variables, or GPU/CPU container setup."
---

# AutoStr Docker Workflow

Use this skill when the container setup is part of the task.

## Goal

Keep the image, compose files, mounted paths, and runtime defaults aligned.

## Workflow

1. Read `Dockerfile` and both compose files before changing runtime behavior.
2. Check which defaults come from the container and which come from Python code.
3. Preserve the fixed container paths for `/input`, `/output`, and `/models`.
4. Keep CPU and GPU settings consistent with the actual runtime choice.
5. Verify that README examples still match the container setup after any change.

## Typical Files To Check

- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.gpu.yml`
- `README.md`
- `main.py`

## When To Use

- You change base images, installed packages, or environment variables.
- You need CPU and GPU builds to behave differently but predictably.
- You are updating the host-to-container mount layout or startup command.

## What Not To Do

- Do not move the fixed container paths unless the whole repo is updated.
- Do not change GPU defaults without checking the compose override.
- Do not leave README examples pointing at an old runtime flow.