import torch
import numpy as np
from typing import Literal


def compute_group_normalized_rewards(
        reward_fn,
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
):
    advantages, raw_rewards = [], []
    for i in range(0, len(rollout_responses), group_size):
        responses = rollout_responses[i:i+group_size]
        ground_truths = repeated_ground_truths[i:i+group_size]
        rewards = []
        for response, ground_truth in zip(responses, ground_truths):
            rewards.append(reward_fn(response, ground_truth)["reward"])
        raw_rewards.extend(rewards)
        rewards_arr = np.array(rewards)
        mean = rewards_arr.mean()
        std = rewards_arr.std(ddof=1)
        rewards_arr -= mean
        if not normalize_by_std:
            advantages.extend(list(rewards_arr))
        else:
            raw_rewards_norm = rewards_arr / (std+advantage_eps)
            advantages.extend(list(raw_rewards_norm))
    return torch.tensor(advantages), torch.tensor(raw_rewards), {}

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor,
                                       policy_log_probs: torch.Tensor) -> torch.Tensor:
    return - raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(advantages: torch.Tensor,
                           policy_log_probs: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           cliprange: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    origin_loss = ratio * advantages
    clip_loss = torch.clip(ratio, min=1-cliprange, max=1+cliprange) * advantages
    grpo_clip_loss = -torch.min(origin_loss, clip_loss)
    return grpo_clip_loss, {"token_clipped": origin_loss > clip_loss}

def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    if loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    if loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata["clip_ratio"] = metadata["token_clipped"].float().mean().item()
        return loss, metadata
    
def masked_mean(tensor: torch.Tensor,
                mask: torch.Tensor,
                dim: int | None = None) -> torch.Tensor:
    masked_tensor = tensor * mask
    total_value = masked_tensor.sum(dim=dim)
    total_count = mask.sum(dim=dim)
    return total_value / total_count

def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )
    loss = masked_mean(total_loss, response_mask, dim=-1).mean() / gradient_accumulation_steps
    loss.backward()
    return loss, metadata