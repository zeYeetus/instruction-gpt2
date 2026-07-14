"""
FastAPI inference backend for an instruction-finetuned GPT-2 small (124M).

Serves the finetuned model as a plain HTTP endpoint that the static
front end (GitHub Pages) calls via fetch(). Uses the same helper
functions the model was finetuned with, so generation matches training.
"""

import torch
import tiktoken
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text

# --- Model configuration -----------------------------------------------------
# GPT-2 small (124M) — matches the retrained weights.
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
}

WEIGHTS_PATH = "instruction_finetuned_gpt2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# --- Load model once at startup ---------------------------------------------
model = GPTModel(BASE_CONFIG)
state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()


def format_input(instruction: str, user_input: str = "") -> str:
    """Alpaca-style prompt, identical to the finetuning format."""
    text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )
    if user_input.strip():
        text += f"\n\n### Input:\n{user_input}"
    return text


# --- API ---------------------------------------------------------------------
app = FastAPI(title="Westley — GPT-2 inference")

# Allow the browser front end (any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenRequest(BaseModel):
    instruction: str
    input: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 1


@app.get("/")
def health():
    return {"status": "ok", "model": "gpt2-small-instruction-finetuned"}


@app.post("/generate")
def do_generate(req: GenRequest):
    instruction = (req.instruction or "").strip()
    if not instruction:
        return {"response": "Enter an instruction to get a response."}

    prompt = format_input(instruction, req.input)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=int(req.max_new_tokens),
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
        temperature=float(req.temperature),
        top_k=int(req.top_k) if int(req.top_k) > 0 else None,
    )
    full = token_ids_to_text(token_ids, tokenizer)

    marker = "### Response:"
    if marker in full:
        answer = full.split(marker, 1)[1].strip()
    else:
        answer = full[len(prompt):].strip()
    return {"response": answer}
