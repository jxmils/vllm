# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from vllm.distributed.trace_context import get_iteration_trace_context
from vllm.utils.torch_utils import record_function_or_nullcontext

from .parallel_state import get_tp_group


def _tensor_shape_label(tensor: torch.Tensor) -> str:
    if tensor.dim() == 0:
        return "-"
    return "x".join(str(dim) for dim in tensor.shape)


def comm_nvtx_label(
    op: str,
    tensor: torch.Tensor,
    **metadata: Any,
) -> str:
    """Build a canonical comm NVTX label.

    ``bytes`` is logical input tensor bytes: ``numel * element_size``.
    """
    ctx = get_iteration_trace_context()
    fields = [
        "comm",
        f"iter={ctx.iteration_id if ctx is not None else -1}",
        f"op={op}",
        f"phase={ctx.phase if ctx is not None else 'unknown'}",
        f"shape={_tensor_shape_label(tensor)}",
        f"bytes={tensor.numel() * tensor.element_size()}",
    ]
    if ctx is not None:
        fields.extend([
            f"rank={ctx.rank}",
            f"pp={ctx.pp}",
            f"tp={ctx.tp}",
        ])
    fields.extend(
        f"{key}={value}" for key, value in metadata.items() if value is not None
    )
    return "|".join(fields)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    with record_function_or_nullcontext(comm_nvtx_label("all_reduce", input_)):
        return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    with record_function_or_nullcontext(
        comm_nvtx_label("all_gather", input_, dim=dim)
    ):
        return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    with record_function_or_nullcontext(
        comm_nvtx_label("reduce_scatter", input_, dim=dim)
    ):
        return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
