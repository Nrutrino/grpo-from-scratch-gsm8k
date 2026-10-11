"""Greedy accuracy on held-out GSM8K test — the real signal (not the rollout reward)."""
import torch

from .reward import is_correct


@torch.no_grad()
def evaluate(model, tok, test_data, cfg, device, n: int | None = None, bs: int | None = None) -> float:
    """
    n=None → the whole test split.  bs defaults to train.eval_batch_size (falls back to
    prompts_per_step).  Generation is greedy, so this is deterministic given the weights.
    """
    model.eval()
    model.config.use_cache = True
    sample = test_data if n is None else test_data[:n]
    bs = bs or cfg["train"].get("eval_batch_size") or cfg["grpo"]["prompts_per_step"]
    correct = 0
    for i in range(0, len(sample), bs):
        chunk   = sample[i: i + bs]
        prompts = [c["prompt"] for c in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=cfg["grpo"]["max_new_tokens"],
            do_sample=False,                       # greedy for eval
            pad_token_id=tok.pad_token_id,
        )
        gen   = out[:, enc.input_ids.shape[1]:]
        comps = tok.batch_decode(gen, skip_special_tokens=True)
        for c, ex in zip(comps, chunk):
            correct += int(is_correct(c, ex["gold"]))
    return correct / max(len(sample), 1)
