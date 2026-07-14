# Instruction-Finetuned GPT-2 — from scratch

A GPT-2 medium (355M) language model implemented from the ground up — token and
positional embeddings, multi-head self-attention, layer normalization, the full
transformer stack — then finetuned on instruction–response pairs so it follows
natural-language tasks. Built following Sebastian Raschka's *Build a Large
Language Model (From Scratch)*.

This repository is the public front end and the inference backend for that model.

## Architecture

The model weights are ~1.4 GB of PyTorch parameters, which can't run in a browser.
So the project is split cleanly:

```
┌────────────────────────┐        POST instruction         ┌─────────────────────────┐
│  Static front end      │  ─────────────────────────────▶ │  Gradio inference API   │
│  (GitHub Pages)        │                                 │  (Hugging Face Space)   │
│  frontend/index.html   │  ◀───────────────────────────── │  space/app.py           │
└────────────────────────┘        generated response       └─────────────────────────┘
```

- **`frontend/`** — a single self-contained `index.html`: Helvetica type, a stone
  palette, and a hand-written WebGL gradient field as the one signature element.
  No build step. Served as-is by GitHub Pages. It calls the Space over HTTP.
- **`space/`** — a Gradio app that loads the finetuned weights, rebuilds the
  GPT-2 medium architecture, and exposes generation as an API. Deployed to a
  Hugging Face Space.

The split is deliberate: it keeps the front end instant and hostable anywhere,
while the model lives where a Python runtime and the weights can sit.

## Deploying

### 1. The model backend (Hugging Face Space)

1. Create a new Space at huggingface.co/new-space, SDK = **Gradio**.
2. Upload `space/app.py`, `space/requirements.txt`, and `space/README.md`.
3. Add the weights with Git LFS (they're too big for a normal commit):
   ```bash
   git lfs install
   git lfs track "*.pth"
   cp /path/to/instruction_finetuned_gpt2.pth .
   git add .gitattributes instruction_finetuned_gpt2.pth
   git commit -m "Add finetuned weights"
   git push
   ```
4. The Space builds and starts. Note its URL, e.g.
   `https://your-username-your-space.hf.space`.

### 2. The front end (GitHub Pages)

1. In `frontend/index.html`, set `SPACE_URL` (near the top of the script) to your
   Space URL from step 1.
2. Push this repo to GitHub.
3. Enable Pages: Settings → Pages → Source = **GitHub Actions**. The included
   workflow publishes `frontend/` on every push to `main`.
4. Your site goes live at `https://your-username.github.io/your-repo/`.

## Notes

- Free Hugging Face Spaces are CPU-only by default; GPT-2 medium takes a few
  seconds per response. Attach a GPU in Space settings for faster inference.
- Generation uses the same helper functions the model was finetuned with, so
  behaviour matches the training notebook exactly.

## License

MIT.
