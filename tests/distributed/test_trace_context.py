# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.communication_op import comm_nvtx_label
from vllm.distributed.trace_context import (
    BatchPhaseInfo,
    batch_phase_nvtx_label,
    get_iteration_trace_context,
    iteration_nvtx_label,
    iteration_trace_context,
    make_iteration_trace_context,
)


def test_iteration_trace_context_labels() -> None:
    phase_info = BatchPhaseInfo(
        phase="prefill",
        prefill_tokens=128,
        decode_tokens=0,
        prefill_reqs=4,
        decode_reqs=0,
        total_tokens=128,
        max_query_len=32,
        num_reqs=4,
    )
    ctx = make_iteration_trace_context(
        37,
        phase_info,
        rank=3,
        pp=0,
        tp=3,
    )

    assert batch_phase_nvtx_label(phase_info) == (
        "vllm:phase=prefill;"
        "prefill_tokens=128;"
        "decode_tokens=0;"
        "prefill_reqs=4;"
        "decode_reqs=0;"
        "total_tokens=128;"
        "max_query_len=32"
    )
    assert iteration_nvtx_label(ctx) == (
        "iter|id=37|phase=prefill|rank=3|pp=0|tp=3|ctx=128|gen=0|"
        "reqs=4|prefill_reqs=4|decode_reqs=0|total_tokens=128|"
        "max_query_len=32"
    )
    assert get_iteration_trace_context() is None

    with iteration_trace_context(ctx):
        tensor = torch.empty((128, 4096), dtype=torch.float16)
        assert get_iteration_trace_context() == ctx
        assert comm_nvtx_label("all_reduce", tensor) == (
            "comm|iter=37|op=all_reduce|phase=prefill|shape=128x4096|"
            "bytes=1048576|rank=3|pp=0|tp=3"
        )

    assert get_iteration_trace_context() is None


def test_comm_nvtx_label_without_context() -> None:
    tensor = torch.empty((), dtype=torch.float32)

    assert comm_nvtx_label("all_gather", tensor, dim=-1) == (
        "comm|iter=-1|op=all_gather|phase=unknown|shape=-|bytes=4|dim=-1"
    )
