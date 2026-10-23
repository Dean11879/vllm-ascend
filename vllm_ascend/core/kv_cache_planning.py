# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Select the upstream planning API without changing the upstream checkout."""

try:
    from vllm.v1.core import kv_cache_planning as planning
except ImportError:
    from vllm.v1.core import kv_cache_utils as planning

_approximate_gcd = planning._approximate_gcd
may_override_num_blocks = planning.may_override_num_blocks
create_kv_cache_group_specs = planning.create_kv_cache_group_specs


def get_kv_cache_groups(*args, **kwargs):
    # Resolve after platform patches have been installed.
    return planning.get_kv_cache_groups(*args, **kwargs)


def unify_hybrid_kv_cache_specs(*args, **kwargs):
    return planning.unify_hybrid_kv_cache_specs(*args, **kwargs)
