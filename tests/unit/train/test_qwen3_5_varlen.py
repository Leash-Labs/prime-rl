import torch

from prime_rl.trainer.model import _flatten_qwen3_5_varlen_batch, _get_qwen3_5_cu_seqlens


def test_flatten_qwen3_5_varlen_batch_preserves_stacked_token_order():
    hidden_states = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    flattened, output_shape = _flatten_qwen3_5_varlen_batch(hidden_states)

    assert flattened.shape == (1, 8, 3)
    assert output_shape == (2, 4)
    torch.testing.assert_close(flattened[0], hidden_states.reshape(-1, 3))
    torch.testing.assert_close(flattened.reshape(*output_shape, -1), hidden_states)


def test_qwen3_5_stacked_rows_ignore_padding_position_resets():
    position_ids = torch.tensor([[0, 1, 2, 0], [0, 1, 0, 0]])

    cu_seqlens = _get_qwen3_5_cu_seqlens(position_ids)

    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 4, 8], dtype=torch.int32))


def test_qwen3_5_single_row_keeps_internal_packed_boundaries():
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    cu_seqlens = _get_qwen3_5_cu_seqlens(position_ids)

    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 3, 5], dtype=torch.int32))


def test_qwen3_5_single_row_collapses_trailing_padding_resets():
    position_ids = torch.tensor([[0, 1, 2, 0, 0, 0]])

    cu_seqlens = _get_qwen3_5_cu_seqlens(position_ids)

    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 3, 6], dtype=torch.int32))
