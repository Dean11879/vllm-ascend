import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.qwen4_exp.fia import QSAFIAMaskWorkspace, build_qsa_fia_mask
from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs
from vllm_ascend.ops.triton.qwen4_exp.qsa import pack_qsa_group_indices_npu

pytestmark = [
    pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]


@torch.inference_mode()
def test_qsa_pack_supports_non_power_of_two_output_width() -> None:
    groups, ratio = 512, 4
    block_indices = torch.arange(groups, dtype=torch.int32).repeat(2, 1).npu()
    query_positions = torch.tensor([2048, 4094], dtype=torch.int64).npu()
    sequence_lengths = torch.tensor([4096, 4096], dtype=torch.int32).npu()

    actual = pack_qsa_group_indices_npu(
        block_indices,
        query_positions,
        sequence_lengths,
        ((query_positions + 1) // ratio).to(torch.int32),
        torch.arange(2, dtype=torch.int32, device="npu"),
        2,
        token_topk=2048,
        compress_ratio=ratio,
    )

    expected = torch.full((2, groups * ratio + ratio - 1), -1, dtype=torch.int32)
    expected[0, :512] = torch.arange(512, dtype=torch.int32)
    expected[0, 512] = 2048
    expected[1, :512] = torch.arange(512, dtype=torch.int32)
    expected[1, 512:515] = torch.arange(4092, 4095, dtype=torch.int32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_pack_masks_unwritten_indexer_slots_and_overwrites_abi() -> None:
    lengths = [1, 2, 3, 4, 15, 2047, 2048, 2049, 2050, 2051]
    indices = torch.full((len(lengths), 512), 777, dtype=torch.int32)
    expected = torch.full((len(lengths), 2051), -1, dtype=torch.int32)
    for row, length in enumerate(lengths):
        count = min(length // 4, 512)
        indices[row, :count] = torch.arange(count - 1, -1, -1, dtype=torch.int32)
        expected[row, :count] = indices[row, :count]
        tail_start = length // 4 * 4
        expected[row, 512 : 512 + length % 4] = torch.arange(tail_start, length, dtype=torch.int32)
    indices = indices.npu()
    row_lengths = torch.tensor(lengths, dtype=torch.int32, device="npu")
    positions = row_lengths.to(torch.int64) - 1
    visible_groups = row_lengths // 4
    storage = torch.full((len(lengths), 4102), -777, dtype=torch.int32, device="npu")
    out = storage[:, ::2]
    for poison in [-777, 12345]:
        out.fill_(poison)
        actual = pack_qsa_group_indices_npu(
            indices,
            positions,
            row_lengths,
            visible_groups,
            torch.zeros(len(lengths), dtype=torch.int32, device="npu"),
            1,
            4,
            2048,
            out,
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert (storage[:, 1::2].cpu() == -777).all()


@torch.inference_mode()
def test_qsa_fia_workspace_wide_to_narrow_reuse() -> None:
    workspace = QSAFIAMaskWorkspace(torch.device("npu:0"), max_rows=2, max_bytes=4096, max_update_width=520)
    compact = torch.full((2, 131), -1, dtype=torch.int32, device="npu")
    compact[0, :2] = torch.tensor([1, 5], dtype=torch.int32, device="npu")
    compact[1, :2] = torch.tensor([2, 9], dtype=torch.int32, device="npu")
    visible = torch.tensor([256, 256], dtype=torch.int32, device="npu")

    actual, selected = build_qsa_fia_mask(compact, visible, kv_length=256, mask_width=256, workspace=workspace)

    expected = torch.ones((2, 256), dtype=torch.bool)
    expected[0, 4:8] = False
    expected[0, 20:24] = False
    expected[1, 8:12] = False
    expected[1, 36:40] = False
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(selected.cpu(), torch.ones(2, dtype=torch.bool), rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_gbsa_preparation_ragged_noncontiguous_table() -> None:
    compact = torch.tensor(
        [[0, 2, -1, 4, -1, -1, -1, -1, 99, 99, 99], [0, 1, 2, 3, 4, 5, 6, 7, 99, 99, 99]],
        dtype=torch.int32,
        device="npu",
    )
    positions = torch.tensor([18, 31], dtype=torch.int64, device="npu")
    lengths = torch.tensor([19, 32], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    table_storage = torch.arange(24, dtype=torch.int32, device="npu").reshape(2, 12)
    block_table = table_storage[:, ::2]

    sparse, counts, visible, cu_query, row_table = prepare_qsa_gbsa_inputs(
        compact, positions, lengths, token_to_req, block_table, num_kv_heads=2
    )

    sparse_cpu = sparse.cpu()
    for head in range(2):
        torch.testing.assert_close(sparse_cpu[head, 0, :3], torch.tensor([0, 2, 4], dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(sparse_cpu[head, 1, :8], torch.arange(8, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), torch.tensor([[3, 8], [3, 8]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(visible.cpu(), torch.tensor([19, 32], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(cu_query.cpu(), torch.tensor([0, 1, 2], dtype=torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(row_table.cpu(), block_table.cpu(), rtol=0, atol=0)
