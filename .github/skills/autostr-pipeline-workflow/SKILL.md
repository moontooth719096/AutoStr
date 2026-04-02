---
name: autostr-pipeline-workflow
description: "Use when changing AutoStr's end-to-end pipeline, batch scan flow, subtitle reflow, or highlight export behavior."
---

# AutoStr Pipeline Workflow

Use this skill for changes that move through the main processing chain.

## Goal

Keep the audio, transcription, alignment, reflow, SRT writing, and highlight export steps consistent with each other.

## Workflow

1. Read `autostr/pipeline.py` first to understand the current orchestration.
2. Check the module-level code that each pipeline step depends on.
3. Verify the expected behavior with the matching tests before editing.
4. Make the smallest change that preserves the existing step order and data flow.
5. Recheck batch mode and highlight mode separately when either path changes.

## Typical Files To Check

- `autostr/pipeline.py`
- `autostr/audio.py`
- `autostr/transcribe.py`
- `autostr/align.py`
- `autostr/reflow.py`
- `autostr/srt_writer.py`
- `autostr/highlight.py`
- `tests/test_pipeline_batch.py`
- `tests/test_pipeline_highlights.py`

## When To Use

- A change affects the overall processing order or output path logic.
- Batch scanning needs to find or skip files differently.
- Highlight export, timing adjustment, or SRT generation needs to stay in sync.

## What Not To Do

- Do not change one stage without checking its downstream output.
- Do not ignore the batch path when the single-file path changes.
- Do not assume highlight mode uses the same output layout as normal subtitle mode.