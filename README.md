# reinforcement_learning_gate1 — GRPO from scratch, gated by a free verifier

A minimal, from-scratch **GRPO** (Group Relative Policy Optimization) implementation
that teaches a small LLM to solve grade-school math (GSM8K), with the policy update
driven entirely by a **free verifier** (answer-checking).

## Why this, why now

The guiding principle (from the journey that led here): **RL cannot create
information — it can only amplify a reward signal into a dense value/discriminator.**
So the only place "compute → capability" genuinely works is where a *cheap, reliable
verifier* exists. Math is the canonical case: the answer is either right or wrong,
checkable for free, infinitely. That free oracle is the entire reason RL works here.

So this repo is deliberately built around the verifier (`src/reward.py`) as the soul,
not the RL algorithm.

## The three pieces that matter (read these)

1. `src/reward.py`  — **the verifier.** Extract the model's answer, compare to gold,
   reward = 1.0 if correct else 0.0. The free oracle.
2. `src/grpo.py`    — **group-relative advantage + the GRPO loss.** For each prompt we
   sample G completions; advantage = how much better than the group average this
   completion's reward is. No value network needed — the *group* is the baseline.
3. `src/train.py`   — **the loop:** sample → verify → advantage → policy-gradient step.

## Method in one paragraph

For each prompt q, sample a group of G completions o₁…o_G from the current policy.
Score each with the verifier → rewards r₁…r_G. The group-relative advantage is
Aᵢ = (rᵢ − mean(r)) / (std(r) + ε): completions that beat their group's average get
pushed up, below-average ones pushed down. The policy is updated with the PPO clipped
surrogate using these advantages, plus a KL penalty toward a frozen reference (the base
model with the LoRA adapter disabled) so it doesn't drift off the rails.

## Setup

```bash
pip install -r requirements.txt
python -m src.train --config config.yaml
```

Model: `Qwen2.5-1.5B-Instruct` + LoRA (single model is BOTH policy [adapter on] and
reference [adapter off] — no second copy). Fits a 24 GB RTX 4090.

## What to watch

- `reward` / `acc` — mean group reward and rollout accuracy. Should climb.
- `kl` — KL to reference. Should stay small (a few × 1e-2); spiking = drifting.
- `len` — completion length. Often grows as the model learns to reason more.
- eval accuracy on held-out GSM8K test every `eval_every` steps — the real signal.
