import torch

from prime_rl.trainer.model import _flatten_qwen3_5_varlen_batch


def test_flatten_qwen3_5_varlen_batch_preserves_stacked_token_order():
    hidden_states = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    flattened, output_shape = _flatten_qwen3_5_varlen_batch(hidden_states)

    assert flattened.shape == (1, 8, 3)
    assert output_shape == (2, 4)
    torch.testing.assert_close(flattened[0], hidden_states.reshape(-1, 3))
    torch.testing.assert_close(flattened.reshape(*output_shape, -1), hidden_states)
