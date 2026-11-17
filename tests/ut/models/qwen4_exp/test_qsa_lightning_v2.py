# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from vllm_ascend.models.qwen4_exp import lightning_indexer as adapter


@pytest.mark.parametrize("query_lengths,kv_lengths", [([2], [25]), ([2, 1, 3], [25, 36, 43])])
@pytest.mark.parametrize("padding", [0, 2])
def test_v2_preserves_request_metadata_and_output_buffer(monkeypatch, query_lengths, kv_lengths, padding):
    requests = len(query_lengths)
    live_rows = sum(query_lengths)
    rows = live_rows + padding
    starts = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist(), *([live_rows] * padding)], dtype=torch.int32
    )
    lengths = torch.tensor(kv_lengths + [0] * padding, dtype=torch.int32)
    owners = torch.repeat_interleave(torch.arange(requests, dtype=torch.int32), torch.tensor(query_lengths))
    owners = torch.cat((owners, torch.zeros(padding, dtype=torch.int32)))
    positions = torch.cat(
        [torch.arange(kv - count, kv, dtype=torch.int32) for count, kv in zip(query_lengths, kv_lengths)]
        + [torch.zeros(padding, dtype=torch.int32)]
    )
    query = torch.empty(rows, 4, 128, dtype=torch.bfloat16)
    cache = torch.empty(8, 32, 1, 128, dtype=query.dtype)
    table = torch.zeros(requests + padding, 2, dtype=torch.int32)
    output = torch.full((rows, 2051), 7, dtype=torch.int32)
    captured = {}

    def metadata(*args, **kwargs):
        captured["metadata"] = kwargs
        return torch.zeros(2048, dtype=torch.int32)

    def indexer(q, k, weights, topk, **kwargs):
        captured["indexer"] = kwargs
        assert q.shape == (live_rows, 4, 128)
        assert k.data_ptr() == cache.data_ptr()
        assert weights.dtype == torch.float32
        expected_weight = torch.tensor(1 / math.sqrt(128), dtype=torch.bfloat16).float()
        assert torch.equal(weights, expected_weight.expand(live_rows, 4))
        assert kwargs["block_table"].shape[0] == requests
        return torch.zeros(live_rows, 1, topk, dtype=torch.int32), torch.empty(0)

    def pack(groups, row_positions, request_lengths, row_owners, ratio, topk, out):
        assert groups.shape == (live_rows, 512)
        assert torch.equal(row_positions, positions[:live_rows])
        assert torch.equal(request_lengths, lengths[:requests])
        assert torch.equal(row_owners, owners[:live_rows])
        assert out.shape == (live_rows, 515)
        out[:, 0] = groups[:, 0]

    monkeypatch.setattr(adapter, "lightning_indexer_v2_metadata", metadata)
    monkeypatch.setattr(adapter, "lightning_indexer_v2", indexer)
    monkeypatch.setattr(adapter, "pack_qsa_group_indices_npu", pack)
    result = adapter.qsa_select_paged_tokens_lightning_v2(
        query,
        cache,
        table,
        owners,
        positions,
        lengths,
        starts,
        2048,
        4,
        output,
        num_actual_requests=requests,
        num_actual_tokens=live_rows,
    )
    assert result is output
    assert (output[:live_rows, 0] == 0).all()
    assert (output[:live_rows, 515:] == -1).all()
    assert (output[live_rows:] == -1).all()
    for kwargs in captured.values():
        assert torch.equal(kwargs["cu_seqlens_q"], starts[: requests + 1])
        assert torch.equal(kwargs["seqused_k"], lengths[:requests] // 4)
        assert torch.equal(kwargs["cmp_residual_k"], lengths[:requests] % 4)
        if requests == 1:
            assert kwargs["cu_seqlens_k"] is None
        else:
            expected = torch.tensor(
                [0, *torch.tensor([n // 4 for n in kv_lengths]).cumsum(0).tolist()], dtype=torch.int32
            )
            assert torch.equal(kwargs["cu_seqlens_k"], expected)
            assert kwargs["cu_seqlens_k"].dtype == torch.int32


def test_v2_empty_batch_skips_operator_calls(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("empty batch must not launch LI V2")

    monkeypatch.setattr(adapter, "lightning_indexer_v2_metadata", unexpected)
    monkeypatch.setattr(adapter, "lightning_indexer_v2", unexpected)
    empty = torch.empty(0, dtype=torch.int32)
    result = adapter.qsa_select_paged_tokens_lightning_v2(
        torch.empty(0, 4, 128, dtype=torch.bfloat16),
        torch.empty(1, 32, 1, 128, dtype=torch.bfloat16),
        torch.empty(0, 1, dtype=torch.int32),
        empty,
        empty,
        empty,
        torch.zeros(1, dtype=torch.int32),
        2048,
        4,
    )
    assert result.shape == (0, 2051)
