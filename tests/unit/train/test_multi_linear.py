import torch

from prime_rl.trainer.models.layers.lora.multi_linear import MultiLoRALinear


def test_single_adapter_disables_grouped_mm():
    layer = MultiLoRALinear(
        torch.nn.Linear(16, 16, bias=False),
        rank=8,
        n_adapters=1,
        use_grouped_mm=True,
    )

    assert layer.use_grouped_mm is False


def test_multiple_aligned_adapters_keep_grouped_mm():
    layer = MultiLoRALinear(
        torch.nn.Linear(16, 16, bias=False),
        rank=8,
        n_adapters=2,
        use_grouped_mm=True,
    )

    assert layer.use_grouped_mm is True
