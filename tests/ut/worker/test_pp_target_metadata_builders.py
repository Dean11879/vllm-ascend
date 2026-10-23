"""Target builders on PP ranks must not initialize the CUDA drafter helper."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.worker.v1.model_runner import NPUModelRunner


@pytest.mark.parametrize("has_drafter", [False, True])
@pytest.mark.parametrize("ubatching", [False, True])
def test_target_metadata_preserves_global_groups_and_defers_drafter(has_drafter, ubatching):
    runner = MagicMock()
    runner.drafter = MagicMock() if has_drafter else None
    runner.parallel_config = SimpleNamespace(use_ubatching=ubatching, num_ubatches=3)
    first, last = MagicMock(), MagicMock()
    runner.attn_groups = [[first], [], [last]]
    config = SimpleNamespace(kv_cache_groups=[object(), object(), object()])

    NPUModelRunner.initialize_metadata_builders(runner, config, [128, 256])

    first.create_metadata_builders.assert_called_once_with(
        runner.vllm_config, runner.device, 128, num_metadata_builders=3 if ubatching else 1
    )
    last.create_metadata_builders.assert_called_once_with(
        runner.vllm_config, runner.device, None, num_metadata_builders=3 if ubatching else 1
    )
    runner.calculate_reorder_batch_threshold.assert_called_once_with()
    if has_drafter:
        runner.drafter.initialize_attn_backend.assert_not_called()
