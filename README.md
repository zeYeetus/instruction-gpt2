# Westley — an instruction-finetuned GPT-2, from scratch

A GPT-2 small (124M) language model implemented from the ground up — token and
positional embeddings, multi-head self-attention, layer normalization, the full
transformer stack — then finetuned on instruction–response pairs so it follows
natural-language tasks. Built following Sebastian Raschka's *Build a Large
Language Model (From Scratch)*. Named after the protagonist of *The Princess Bride*.

This repository is the public front end and the inference backend for that model.

## Architecture

The model can't run in a browser, so the project is split cleanly:

```
┌────────────────────────┐        POST /generate           ┌─────────────────────────┐
│  Static front end      │  ─────────────────────────────▶ │  FastAPI inference API  │
│  (GitHub Pages)        │                                 │  (Render web service)   │
│  frontend/index.html   │  ◀───────────────────────────── │  space/app.py           │
└────────────────────────┘        { "response": ... }      └─────────────────────────┘
```

- **`frontend/`** — a single self-contained `index.html`: Helvetica type, a
  phthalo-blue moving WebGL gradient, and liquid-glass panels. No build step;
  served as-is by GitHub Pages. Calls the backend over HTTP.
- **`space/`** — a FastAPI app that loads the finetuned weights, rebuilds the
  GPT-2 small architecture, and serves generation at `POST /generate`.
- **`render.yaml`** — Render blueprint that builds and runs the backend.

## Deploying

### Backend (Render)
1. Push this repo to GitHub.
2. On render.com, create a new **Web Service** from the repo (or use the
   `render.yaml` blueprint). Root directory `space`, free plan.
3. Add the weights via Git LFS before pushing (too big for a normal commit):
   ```bash
   git lfs install
   git lfs track "*.pth"
   cp /path/to/instruction_finetuned_gpt2.pth space/
   git add .gitattributes space/instruction_finetuned_gpt2.pth
   git commit -m "Add finetuned weights"
   git push
   ```
4. Render builds and starts it. Note the URL, e.g. `https://westley-model.onrender.com`.

### Front end (GitHub Pages)
1. In `frontend/index.html`, set `API_URL` to your Render URL from above.
2. Push. Settings → Pages → Source = **GitHub Actions**. The workflow publishes
   `frontend/` on every push.
3. Live at `https://your-username.github.io/your-repo/`.

## Notes
- Render's free tier sleeps when idle; the first request after idle takes ~30–60s
  to wake the service (the front end shows a "waking the model" message). Later
  requests are faster.
- Generation uses the same helper functions the model was finetuned with, so
  behaviour matches the training notebook.

## License
MIT.
