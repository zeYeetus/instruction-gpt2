"""
Backend inference API for an instruction-finetuned GPT-2 medium.

This wraps the exact helper functions used during finetuning
(from the llms-from-scratch package) so generation behaviour matches
the training notebook. It exposes a small Gradio app whose /api/predict
endpoint the static front end calls via fetch().
"""

import torch
import tiktoken
import gradio as gr

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text

# --- Model configuration -----------------------------------------------------
# Must match the architecture the weights were trained with (GPT-2 medium).
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
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


def respond(instruction: str, user_input: str = "",
            max_new_tokens: int = 256, temperature: float = 0.0,
            top_k: int = 1) -> str:
    instruction = (instruction or "").strip()
    if not instruction:
        return "Enter an instruction to get a response."

    prompt = format_input(instruction, user_input)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=int(max_new_tokens),
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
        temperature=float(temperature),
        top_k=int(top_k) if int(top_k) > 0 else None,
    )
    full = token_ids_to_text(token_ids, tokenizer)

    # Return only the model's completion, not the echoed prompt.
    marker = "### Response:"
    if marker in full:
        return full.split(marker, 1)[1].strip()
    return full[len(prompt):].strip()


demo = gr.Interface(
    fn=respond,
    inputs=[
        gr.Textbox(label="Instruction", lines=2,
                   placeholder="Rewrite the sentence in the passive voice."),
        gr.Textbox(label="Input (optional)", lines=2,
                   placeholder="The committee approved the proposal."),
        gr.Slider(16, 512, value=256, step=16, label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="Temperature"),
        gr.Slider(0, 50, value=1, step=1, label="Top-k (0 = greedy)"),
    ],
    outputs=gr.Textbox(label="Response", lines=6),
    title="Instruction-Finetuned GPT-2",
    description="A GPT-2 medium model finetuned to follow instructions, "
                "built from scratch following Raschka's LLMs-from-scratch.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
