"""
GRPO training loop:  sample → verify → group-advantage → policy-gradient step.

    python -m src.train --config config.yaml
"""
import os
import csv
import json
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


def save_ckpt(model, optimizer, step, path):
    """LoRA adapter + optimizer state + step → enough to resume after an interruption."""
    model.save_pretrained(path)
    torch.save({"optimizer": optimizer.state_dict(), "step": step},
               os.path.join(path, "trainer_state.pt"))


def load_ckpt(model, optimizer, path, device):
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, load_file(os.path.join(path, "adapter_model.safetensors")))
    state = torch.load(os.path.join(path, "trainer_state.pt"), map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    return state["step"]


def train(cfg_path: str, resume: str | None = None):
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
    log_dir = t.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    start_step = 0
    if resume:
        start_step = load_ckpt(model, optimizer, resume, device)
        set_seed(t["seed"] + start_step)        # fresh-but-reproducible sampling after resume
        print(f"[resume] {resume} → continuing from step {start_step}")
    mode = "a" if resume else "w"               # append to the logs when resuming
    train_f = open(os.path.join(log_dir, "train_log.csv"), mode, newline="", encoding="utf-8")
    eval_f  = open(os.path.join(log_dir, "eval_log.csv"),  mode, newline="", encoding="utf-8")
    train_csv, eval_csv = csv.writer(train_f), csv.writer(eval_f)
    if not resume:
        train_csv.writerow(["step", "reward", "acc", "loss", "kl", "clip_frac", "len", "lr", "sec"])
        eval_csv.writerow(["step", "test_acc", "n"])
    results = {"model": cfg["model"]["name"], "lr": t["lr"], "total_steps": t["total_steps"]}

    def log_eval(step, n, tag=None):
        acc_ = evaluate(model, tok, test_data, cfg, device, n)
        n_ = len(test_data) if n is None else min(n, len(test_data))
        eval_csv.writerow([step, f"{acc_:.4f}", n_]); eval_f.flush()
        print(f"  [eval{'/' + tag if tag else ''}] test acc @ step {step} = {acc_:.1%}  (n={n_})")
        return acc_

    # ── Step-0 baseline on the FULL test split: without it "improvement" is unmeasurable ──
    # Greedy decoding is deterministic, so a baseline measured earlier can be reused via
    # train.baseline_acc (config) instead of re-running the ~20 min full eval.
    if resume:
        results["baseline_acc"] = t.get("baseline_acc")
    elif t.get("baseline_acc") is not None:
        results["baseline_acc"] = t["baseline_acc"]
        eval_csv.writerow([0, f"{t['baseline_acc']:.4f}", len(test_data)]); eval_f.flush()
        print(f"  [eval/baseline] reused: test acc @ step 0 = {t['baseline_acc']:.1%}")
    elif t.get("eval_full_baseline", True):
        results["baseline_acc"] = log_eval(0, None, "baseline")

    hdr = (f"{'step':>5} │ {'reward':>6} │ {'acc':>5} │ {'loss':>8} │ {'kl':>7} │ "
           f"{'clip%':>6} │ {'len':>5} │ {'lr':>8} │ {'time':>5}")

    for step in range(start_step, t["total_steps"]):
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

        # Whole-batch metrics (last inner epoch): loss chunks are already scaled by the global
        # token count so they sum to the full-batch loss; kl / clip are token-weighted means.
        tot_loss = tot_kl = tot_clip = 0.0
        for _ in range(g["inner_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            tot_loss = tot_kl = tot_clip = 0.0
            for i in range(0, seqs.shape[0], micro_bs):
                sl = slice(i, i + micro_bs)
                new_logp = token_logprobs(model, seqs[sl], attn[sl])     # WITH grad
                loss, kl, clipf = grpo_loss(
                    new_logp, old_logp[sl], ref_logp[sl], adv[sl], cmask[sl],
                    clip_eps=g["clip_eps"], kl_coef=g["kl_coef"], token_norm=token_norm,
                )
                loss.backward()                                          # accumulate
                w = cmask[sl].sum().item() / max(token_norm, 1.0)
                tot_loss += loss.item()
                tot_kl   += kl.item() * w
                tot_clip += clipf.item() * w
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            optimizer.step()

        # ── 6. Log ────────────────────────────────────────────────────────────
        mean_len = cmask.sum().item() / seqs.shape[0]
        sec = time.time() - t0
        train_csv.writerow([step, f"{rewards.mean().item():.4f}", f"{acc:.4f}", f"{tot_loss:.5f}",
                            f"{tot_kl:.6f}", f"{tot_clip:.4f}", f"{mean_len:.1f}", f"{lr:.3e}", f"{sec:.1f}"])
        train_f.flush()
        if step % t["log_every"] == 0:
            if step % (t["log_every"] * 20) == 0:
                print(hdr); print("─" * len(hdr))
            print(f"{step:>5} │ {rewards.mean().item():>6.3f} │ {acc:>5.1%} │ "
                  f"{tot_loss:>8.4f} │ {tot_kl:>7.4f} │ {tot_clip:>6.1%} │ "
                  f"{mean_len:>5.0f} │ {lr:>8.1e} │ {sec:>4.0f}s")

        # ── 7. Eval / save ────────────────────────────────────────────────────
        is_last = (step + 1) == t["total_steps"]
        if (step + 1) % t["eval_every"] == 0 and not (is_last and t.get("eval_full_final", True)):
            log_eval(step + 1, t["eval_n"])
        if (step + 1) % t["save_every"] == 0:
            path = os.path.join(t["ckpt_dir"], f"lora_step{step+1}")
            save_ckpt(model, optimizer, step + 1, path)
            print(f"  [ckpt] → {path}")

    model.save_pretrained(os.path.join(t["ckpt_dir"], "lora_final"))
    if t.get("eval_full_final", True):
        results["final_acc"] = log_eval(t["total_steps"], None, "final")
    with open(os.path.join(log_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    train_f.close(); eval_f.close()
    print(f"[done] {results}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint dir, e.g. checkpoints/lora_step100")
    a = ap.parse_args()
    train(a.config, a.resume)
