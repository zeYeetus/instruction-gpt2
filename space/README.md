---
title: Instruction-Finetuned GPT-2
emoji: ◐
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Instruction-Finetuned GPT-2 — inference API

A GPT-2 medium (355M) model built from scratch and instruction-finetuned,
following Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

This Space serves the model as an inference backend. The public front end
that consumes it lives at the project's GitHub Pages site.

## Files
- `app.py` — loads the finetuned weights and exposes generation via Gradio.
- `requirements.txt` — Python dependencies.
- `instruction_finetuned_gpt2.pth` — the finetuned weights (tracked with Git LFS).

## Calling it from the front end
The Gradio app exposes an HTTP API. The static site posts an instruction to
the `/call/predict` endpoint and reads back the generated response.
