from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty
        self.turn_reward_metrics = config.turn_reward_metrics

    @staticmethod
    def _assign_turn_advantages(rollout: Rollout, turn_advantages: list[float]) -> None:
        """Align one group-relative advantage per sampled turn to trainable tokens."""
        sampled_nodes = [node for node in rollout.nodes if node.sampled]
        if len(sampled_nodes) != len(turn_advantages):
            raise ValueError(
                "rollout sampled-turn count does not match configured metrics "
                f"({len(sampled_nodes)} != {len(turn_advantages)})"
            )
        advantages_by_node = {
            id(node): advantage for node, advantage in zip(sampled_nodes, turn_advantages, strict=True)
        }
        branches = list(iter_trainable_branches(rollout))
        if len(branches) != len(rollout.samples):
            raise ValueError(
                f"turn-aware GRPO branch/sample mismatch: {len(branches)} branches, {len(rollout.samples)} samples"
            )

        values: list[float] = []
        for (branch, trainable_mask), sample in zip(branches, rollout.samples, strict=True):
            if len(trainable_mask) != len(sample.token_ids):
                raise ValueError(
                    "turn-aware GRPO mask/sample mismatch: "
                    f"{len(trainable_mask)} mask values, {len(sample.token_ids)} tokens"
                )
            sample_values = [0.0] * len(sample.token_ids)
            offset = 0
            for node in branch.nodes:
                node_length = len(node.token_ids)
                if node.sampled:
                    advantage = advantages_by_node[id(node)]
                    for index, trainable in enumerate(trainable_mask[offset : offset + node_length]):
                        if trainable:
                            sample_values[offset + index] = advantage
                offset += node_length
            values.extend(sample_values)
        rollout.assign_advantages(values)

    async def score_group(self, group: list[Rollout]) -> None:
        if self.turn_reward_metrics:
            missing = {name for rollout in group for name in self.turn_reward_metrics if name not in rollout.metrics}
            if missing:
                raise ValueError("turn-aware GRPO rollout metrics are missing: " + ", ".join(sorted(missing)))
            rewards = torch.tensor(
                [[rollout.metrics[name] for name in self.turn_reward_metrics] for rollout in group],
                dtype=torch.float32,
            )
            advantages = rewards - rewards.mean(dim=0, keepdim=True)
            for rollout, per_turn in zip(group, advantages.tolist(), strict=True):
                self._assign_turn_advantages(rollout, per_turn)
            return

        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        length_penalty = self.length_penalty
        if length_penalty is None:
            advantages = rewards - rewards.mean()
        else:
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
            advantages = shaped_rewards - shaped_rewards.mean()
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
