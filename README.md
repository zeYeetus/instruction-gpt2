# Westley — an instruction-following GPT-2, built from scratch

Westley is a GPT-2 small (124M) language model implemented from the ground up:
token and positional embeddings, multi-head self-attention, layer normalization,
and the full transformer stack written by hand then finetuned on
instruction–response pairs so it follows natural-language tasks. It is built
following Sebastian Raschka's *Build a Large Language Model (From Scratch)* and
Vizuara's *Building LLMs from scratch*, and named after the protagonist of
*The Princess Bride*.

This repository contains the model, the inference backend, and a web front end.

## Run it locally

The front end is hosted live (see below), but the model runs locally. To try the
full thing end to end:

1. Get the code and the weights. The weights are stored with Git LFS, so install
   that first, then clone:

```
git lfs install
git clone https://github.com/zeYeetus/Westley-LLM.git
cd Westley-LLM
```

2. Start the backend:

```
cd space
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete`.

3. Open the front end. Open `frontend/index.html` in a browser, type an
   instruction (e.g. "Rewrite this sentence in the passive voice." with input
   "The committee approved the proposal."), and hit Generate. The page calls your
   local backend and shows the model's response.

## Live front end

The web UI is deployed on GitHub Pages:
**https://zeyeetus.github.io/Westley-LLM/**

The Generate button there is wired to a local backend, so it's interactive only
when you're running `space/app.py` on your own machine (per the steps above).
The page explains this when the backend isn't reachable.

## License

MIT. Model code adapted from Sebastian Raschka's LLMs-from-scratch (Apache 2.0).
