# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from vllm.utils.torch_utils import record_function_or_nullcontext

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    with record_function_or_nullcontext(
        "vllm:comm:tp_all_reduce;"
        f"shape={tuple(input_.shape)};"
        f"bytes={input_.numel() * input_.element_size()}"
    ):
        return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    with record_function_or_nullcontext(
        "vllm:comm:tp_all_gather;"
        f"shape={tuple(input_.shape)};"
        f"bytes={input_.numel() * input_.element_size()};"
        f"dim={dim}"
    ):
        return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    with record_function_or_nullcontext(
        "vllm:comm:tp_reduce_scatter;"
        f"shape={tuple(input_.shape)};"
        f"bytes={input_.numel() * input_.element_size()};"
        f"dim={dim}"
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
