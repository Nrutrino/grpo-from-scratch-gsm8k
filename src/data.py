"""
GSM8K loading + prompt construction.

Each example becomes a chat prompt (Qwen ChatML) asking for step-by-step reasoning
ending in \\boxed{ANSWER}, plus the parsed gold number for the verifier.
"""
from datasets import load_dataset

from .reward import extract_gold

SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step. "
    "On the final line, give ONLY the final answer in the form \\boxed{ANSWER}."
)


def build_prompt(tokenizer, question: str) -> str:
    """Render a question into a Qwen ChatML prompt string (generation-ready)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_split(cfg, tokenizer, split: str):
    """
    Returns a list of dicts: {"prompt": str, "gold": str, "question": str}.
    split: "train" or "test".
    """
    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["subset"], split=split)
    out = []
    for ex in ds:
        gold = extract_gold(ex["answer"])
        if gold is None:
            continue
        out.append({
            "prompt":   build_prompt(tokenizer, ex["question"]),
            "gold":     gold,
            "question": ex["question"],
        })
    return out
