# Westley — inference backend

A FastAPI service that serves an instruction-finetuned GPT-2 small (124M),
built from scratch following Sebastian Raschka's *Build a Large Language
Model (From Scratch)*.

## Files
- `app.py` — loads the finetuned weights and exposes `POST /generate`.
- `requirements.txt` — Python dependencies.
- `instruction_finetuned_gpt2.pth` — the finetuned weights (tracked with Git LFS).

## Endpoint
`POST /generate` with JSON:
```json
{ "instruction": "...", "input": "", "max_new_tokens": 256, "temperature": 0.0, "top_k": 1 }
```
returns `{ "response": "..." }`.

`GET /` is a health check.

## Running locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then POST to http://localhost:8000/generate.

## Deployed on Render
This service is deployed as a free Render web service (see `render.yaml` in the
repo root). The static front end on GitHub Pages calls it via `fetch()`.
