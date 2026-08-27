from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import LakatosGRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class LakatosGRPOAlgorithm(GRPOAlgorithm):
    def __init__(self, config: LakatosGRPOAlgoConfig, policy_pool: InferencePool):
        Algorithm.__init__(self, config, policy_pool)
        self.reward_metric_prefix = config.reward_metric_prefix

    def _rewards(self, group: list[Rollout]) -> torch.Tensor:
        turn_counts = {rollout.num_turns for rollout in group}
        if len(turn_counts) != 1:
            raise ValueError("lakatos_grpo requires equal turn counts within a rollout group")
        turn_count = turn_counts.pop()
        names = [f"{self.reward_metric_prefix}{index}" for index in range(turn_count)]
        missing = {name for rollout in group for name in names if name not in rollout.metrics}
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"lakatos_grpo rollout metrics are missing: {joined}")
        return torch.tensor(
            [[rollout.metrics[name] for name in names] for rollout in group],
            dtype=torch.float32,
        )

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = self._rewards(group)
        returns = rewards.flip(1).cumsum(1).flip(1)
        centered = returns - returns.mean(dim=0, keepdim=True)
        standard_deviation = returns.std(dim=0, correction=0, keepdim=True)
        advantages = torch.where(
            standard_deviation > 0.0,
            centered / standard_deviation.clamp(min=torch.finfo(returns.dtype).eps),
            torch.zeros_like(centered),
        )
        for rollout, per_turn in zip(group, advantages.tolist(), strict=True):
            self._assign_turn_advantages(rollout, per_turn)
