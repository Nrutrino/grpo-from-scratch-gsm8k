"""
The GRPO core: group-relative advantage + the clipped policy-gradient loss.

GRPO's trick: instead of learning a value network as the baseline (like PPO), it uses
the *group itself* as the baseline.  Sample G completions for one prompt, and a
completion's advantage is simply how far its reward sits above/below the group mean,
normalized by the group's spread.  Cheap, value-network-free, and the reason it fits
a single 4090.
"""
import torch


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """
    rewards: [B] where B = num_prompts × group_size, ordered grouped-by-prompt.
    Returns advantages [B]: (r − group_mean) / (group_std + eps), one scalar per completion.

    A group where every completion got the same reward (all right or all wrong) has
    std=0 → advantage≈0 → no gradient from it.  That is correct: it taught us nothing.
    """
    r = rewards.view(-1, group_size)                       # [P, G]
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.reshape(-1)                                 # [B]


def grpo_loss(
    new_logp:  torch.Tensor,   # [b, L-1]  log π_θ      (WITH grad)
    old_logp:  torch.Tensor,   # [b, L-1]  log π_θ_old  (rollout-time, no grad)
    ref_logp:  torch.Tensor,   # [b, L-1]  log π_ref    (adapter off, no grad)
    adv:       torch.Tensor,   # [b]       per-sequence advantage
    mask:      torch.Tensor,   # [b, L-1]  completion-token mask
    clip_eps:  float,
    kl_coef:   float,
    token_norm: float,         # global #completion-tokens in the full batch (for grad-accum)
):
    """
    PPO clipped surrogate with group-relative advantages + KL-to-reference.

      ratio   = exp(log π_θ − log π_θ_old)
      surr    = min(ratio·A, clip(ratio, 1±ε)·A)            # don't move too far per step
      KL(θ‖ref) ≈ exp(r−p) − (r−p) − 1   with (r−p)=ref−new # k3 unbiased estimator, ≥0
      loss    = −surr + β·KL,   averaged over completion tokens

    `token_norm` lets us micro-batch the update with gradient accumulation while keeping
    the SAME normalization as a single full-batch mean (divide each chunk by the global
    token count, sum the backward passes).
    """
    ratio = torch.exp(new_logp - old_logp)                 # [b, L-1]
    a     = adv.unsqueeze(1)                                # [b, 1] → broadcast over tokens
    surr  = torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)

    diff = ref_logp - new_logp
    kl   = torch.exp(diff) - diff - 1.0                    # [b, L-1], ≥ 0

    per_tok = -(surr) + kl_coef * kl                       # [b, L-1]
    loss = (per_tok * mask).sum() / max(token_norm, 1.0)

    with torch.no_grad():
        mean_kl = (kl * mask).sum() / mask.sum().clamp(min=1.0)
        clipped = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float()
        clip_frac = (clipped * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, mean_kl, clip_frac
