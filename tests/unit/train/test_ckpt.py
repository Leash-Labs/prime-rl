import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prime_rl.trainer.ckpt import restore_scheduler_state


def _make_optimizer_and_scheduler():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-4)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=590, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])
    return optimizer, scheduler


def test_restore_scheduler_state_reapplies_optimizer_learning_rate():
    optimizer, scheduler = _make_optimizer_and_scheduler()
    for _ in range(101):
        optimizer.step()
        scheduler.step()

    state_dict = scheduler.state_dict()
    expected_lr = scheduler.get_last_lr()[0]

    resumed_optimizer, resumed_scheduler = _make_optimizer_and_scheduler()
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    restore_scheduler_state(resumed_scheduler, state_dict)

    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
