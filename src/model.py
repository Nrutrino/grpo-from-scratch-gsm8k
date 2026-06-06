"""
Model: a single Qwen2.5-1.5B-Instruct + LoRA that serves as BOTH
  • the policy   (adapter enabled), and
  • the reference (adapter disabled via `model.disable_adapter()`),
so we never hold a second copy in VRAM.

Plus the two low-level helpers GRPO needs:
  • generate_group()  — sample G completions per prompt.
  • token_logprobs()  — per-token log π(token | prefix), correct under LEFT padding.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_policy(cfg, device):
    name  = cfg["model"]["name"]
    dtype = _DTYPE[cfg["model"].get("dtype", "bfloat16")]

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left-pad so generation continues at a uniform column

    base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    base.config.use_cache = True
    lora = LoraConfig(
        r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    return model, tok


@torch.no_grad()
def generate_group(model, tok, prompt_texts, cfg, device):
    """
    Sample G completions for each prompt.

    Returns
        seqs        [B, L]  full token ids (left-padded prompt + completion), B = N×G
        attn        [B, L]  1 for real tokens, 0 for pad
        prompt_len  int     padded prompt length (completion starts at this column)
    """
    G = cfg["grpo"]["group_size"]
    enc = tok(prompt_texts, return_tensors="pt", padding=True).to(device)
    prompt_len = enc.input_ids.shape[1]

    model.eval()
    model.config.use_cache = True   # token_logprobs() disables it; generation needs it
    seqs = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=cfg["grpo"]["max_new_tokens"],
        do_sample=True,
        temperature=cfg["grpo"]["temperature"],
        top_p=cfg["grpo"]["top_p"],
        num_return_sequences=G,        # → rows grouped by prompt: [p0×G, p1×G, ...]
        pad_token_id=tok.pad_token_id,
    )
    attn = (seqs != tok.pad_token_id).long()
    return seqs, attn, prompt_len


def _position_ids(attention_mask):
    """Correct RoPE positions for LEFT-padded sequences (cumsum of the mask)."""
    pos = attention_mask.long().cumsum(-1) - 1
    pos.masked_fill_(attention_mask == 0, 0)   # pad positions are masked out anyway
    return pos


def token_logprobs(model, input_ids, attention_mask):
    """
    Per-token log-prob of each *actual* token given its prefix.

    Returns logp [B, L-1] aligned so logp[:, j] = log π(input_ids[:, j+1] | ≤ j).
    Caller masks with the completion mask (see completion_mask()).
    """
    model.config.use_cache = False
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=_position_ids(attention_mask),
    )
    logits  = out.logits[:, :-1, :]            # predicts tokens 1..L-1
    targets = input_ids[:, 1:]                 # [B, L-1]
    logp    = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [B, L-1]


def completion_mask(attention_mask, prompt_len):
    """
    Mask aligned to token_logprobs() output [B, L-1]: 1 only on real COMPLETION
    tokens (excludes the prompt and any right-padding after EOS).
    """
    mask = attention_mask[:, 1:].clone().float()   # target at col j is input[:, j+1]
    mask[:, : prompt_len - 1] = 0.0                # zero out prompt-target positions
    return mask
