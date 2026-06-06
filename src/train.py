"""
GRPO training loop:  sample → verify → group-advantage → policy-gradient step.

    python -m src.train --config config.yaml
"""
import os
import time
import random
import argparse
import contextlib

import yaml
import torch

from .model import (load_policy, generate_group, token_logprobs, completion_mask)
from .grpo import group_advantages, grpo_loss
from .reward import compute_reward, is_correct
from .data import load_split
from .eval import evaluate


def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def batched_logprobs(model, seqs, attn, micro_bs, adapter_on: bool):
    """log π over all sequences, micro-batched, no grad.  adapter_on=False → reference."""
    model.eval()
    ctx = contextlib.nullcontext() if adapter_on else model.disable_adapter()
    outs = []
    with ctx:
        for i in range(0, seqs.shape[0], micro_bs):
            outs.append(token_logprobs(model, seqs[i:i + micro_bs], attn[i:i + micro_bs]))
    return torch.cat(outs, dim=0)


def lr_at(step, base_lr, warmup):
    return base_lr * min(1.0, (step + 1) / max(1, warmup))


def train(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    g = cfg["grpo"]
    t = cfg["train"]
    G, N      = g["group_size"], g["prompts_per_step"]
    micro_bs  = g["micro_batch_size"]
    fmt_bonus = g["format_bonus"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(t["seed"])

    model, tok = load_policy(cfg, device)
    train_data = load_split(cfg, tok, "train")
    test_data  = load_split(cfg, tok, "test")
    print(f"[data] {len(train_data):,} train | {len(test_data):,} test")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=0.0)

    os.makedirs(t["ckpt_dir"], exist_ok=True)
    hdr = (f"{'step':>5} │ {'reward':>6} │ {'acc':>5} │ {'loss':>8} │ {'kl':>7} │ "
           f"{'clip%':>6} │ {'len':>5} │ {'lr':>8} │ {'time':>5}")

    for step in range(t["total_steps"]):
        t0 = time.time()

        # ── 1. Sample a batch of prompts ──────────────────────────────────────
        batch   = random.sample(train_data, N)
        prompts = [b["prompt"] for b in batch]
        golds   = [b["gold"]   for b in batch]

        # ── 2. Rollout: G completions per prompt (rows grouped by prompt) ──────
        seqs, attn, prompt_len = generate_group(model, tok, prompts, cfg, device)
        comps = tok.batch_decode(seqs[:, prompt_len:], skip_special_tokens=True)
        golds_exp = [golds[i] for i in range(N) for _ in range(G)]   # gold per completion

        # ── 3. Verify → reward → group-relative advantage ─────────────────────
        rewards = torch.tensor(
            [compute_reward(c, gd, fmt_bonus) for c, gd in zip(comps, golds_exp)],
            dtype=torch.float32, device=device,
        )
        acc = sum(is_correct(c, gd) for c, gd in zip(comps, golds_exp)) / len(comps)
        adv   = group_advantages(rewards, G)
        cmask = completion_mask(attn, prompt_len)
        token_norm = cmask.sum().item()

        # ── 4. Fixed reference points: old (policy@rollout) + ref (adapter off) ─
        old_logp = batched_logprobs(model, seqs, attn, micro_bs, adapter_on=True)
        ref_logp = batched_logprobs(model, seqs, attn, micro_bs, adapter_on=False)

        # ── 5. Policy-gradient update (micro-batched grad accumulation) ────────
        model.train()
        lr = lr_at(step, t["lr"], t["warmup_steps"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        last_kl = last_clip = 0.0
        for _ in range(g["inner_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, seqs.shape[0], micro_bs):
                sl = slice(i, i + micro_bs)
                new_logp = token_logprobs(model, seqs[sl], attn[sl])     # WITH grad
                loss, kl, clipf = grpo_loss(
                    new_logp, old_logp[sl], ref_logp[sl], adv[sl], cmask[sl],
                    clip_eps=g["clip_eps"], kl_coef=g["kl_coef"], token_norm=token_norm,
                )
                loss.backward()                                          # accumulate
                last_kl, last_clip = kl.item(), clipf.item()
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            optimizer.step()

        # ── 6. Log ────────────────────────────────────────────────────────────
        mean_len = cmask.sum().item() / seqs.shape[0]
        if step % t["log_every"] == 0:
            if step % (t["log_every"] * 20) == 0:
                print(hdr); print("─" * len(hdr))
            print(f"{step:>5} │ {rewards.mean().item():>6.3f} │ {acc:>5.1%} │ "
                  f"{loss.item():>8.4f} │ {last_kl:>7.4f} │ {last_clip:>6.1%} │ "
                  f"{mean_len:>5.0f} │ {lr:>8.1e} │ {time.time()-t0:>4.0f}s")

        # ── 7. Eval / save ────────────────────────────────────────────────────
        if (step + 1) % t["eval_every"] == 0:
            test_acc = evaluate(model, tok, test_data, cfg, device, t["eval_n"])
            print(f"  [eval] test acc @ step {step+1} = {test_acc:.1%}  (n={t['eval_n']})")
        if (step + 1) % t["save_every"] == 0:
            path = os.path.join(t["ckpt_dir"], f"lora_step{step+1}")
            model.save_pretrained(path)
            print(f"  [ckpt] → {path}")

    model.save_pretrained(os.path.join(t["ckpt_dir"], "lora_final"))
    print("[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    train(ap.parse_args().config)
