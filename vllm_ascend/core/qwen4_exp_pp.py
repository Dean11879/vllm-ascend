# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""The PLE ownership contract supported by the Ascend PP runners."""

from vllm.distributed.utils import get_pp_indices


def validate_ple_pipeline_partition(text_config, parallel_config) -> None:
    """Require raw-token PLE work to remain on the first pipeline stage.

    PLE layer IDs in the checkpoint are one-based. Respect the same explicit
    VLLM_PP_LAYER_PARTITION as make_layers, including uneven partitions.
    """
    if not text_config.ple_layer_ids or parallel_config.pipeline_parallel_size == 1:
        return
    _, first_stage_end = get_pp_indices(text_config.num_hidden_layers, 0, parallel_config.pipeline_parallel_size)
    outside = [layer for layer in text_config.ple_layer_ids if not 1 <= layer <= first_stage_end]
    if outside:
        raise NotImplementedError(
            "Ascend Qwen4Exp PP requires all PLE layers on the first pipeline stage; "
            f"one-based PLE layers {outside} fall outside its layers 1..{first_stage_end}. "
            "Choose a compatible VLLM_PP_LAYER_PARTITION or use PP=1."
        )
