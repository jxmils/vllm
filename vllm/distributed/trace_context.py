# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = (
    "BatchPhaseInfo",
    "IterationTraceContext",
    "batch_phase_nvtx_label",
    "get_iteration_trace_context",
    "iteration_nvtx_label",
    "iteration_trace_context",
    "make_iteration_trace_context",
)


@dataclass(frozen=True)
class BatchPhaseInfo:
    phase: str
    prefill_tokens: int
    decode_tokens: int
    prefill_reqs: int
    decode_reqs: int
    total_tokens: int
    max_query_len: int
    num_reqs: int


@dataclass(frozen=True)
class IterationTraceContext:
    iteration_id: int
    phase: str
    rank: int
    pp: int
    tp: int
    prefill_tokens: int
    decode_tokens: int
    prefill_reqs: int
    decode_reqs: int
    total_tokens: int
    max_query_len: int
    num_reqs: int


_ITERATION_TRACE_CONTEXT: ContextVar[IterationTraceContext | None] = ContextVar(
    "vllm_iteration_trace_context",
    default=None,
)


def batch_phase_nvtx_label(info: BatchPhaseInfo) -> str:
    return (
        f"vllm:phase={info.phase};"
        f"prefill_tokens={info.prefill_tokens};"
        f"decode_tokens={info.decode_tokens};"
        f"prefill_reqs={info.prefill_reqs};"
        f"decode_reqs={info.decode_reqs};"
        f"total_tokens={info.total_tokens};"
        f"max_query_len={info.max_query_len}"
    )


def iteration_nvtx_label(ctx: IterationTraceContext) -> str:
    return (
        f"iter|id={ctx.iteration_id}"
        f"|phase={ctx.phase}"
        f"|rank={ctx.rank}"
        f"|pp={ctx.pp}"
        f"|tp={ctx.tp}"
        f"|ctx={ctx.prefill_tokens}"
        f"|gen={ctx.decode_tokens}"
        f"|reqs={ctx.num_reqs}"
        f"|prefill_reqs={ctx.prefill_reqs}"
        f"|decode_reqs={ctx.decode_reqs}"
        f"|total_tokens={ctx.total_tokens}"
        f"|max_query_len={ctx.max_query_len}"
    )


def make_iteration_trace_context(
    iteration_id: int,
    phase_info: BatchPhaseInfo,
    *,
    rank: int,
    pp: int,
    tp: int,
) -> IterationTraceContext:
    return IterationTraceContext(
        iteration_id=iteration_id,
        phase=phase_info.phase,
        rank=rank,
        pp=pp,
        tp=tp,
        prefill_tokens=phase_info.prefill_tokens,
        decode_tokens=phase_info.decode_tokens,
        prefill_reqs=phase_info.prefill_reqs,
        decode_reqs=phase_info.decode_reqs,
        total_tokens=phase_info.total_tokens,
        max_query_len=phase_info.max_query_len,
        num_reqs=phase_info.num_reqs,
    )


def get_iteration_trace_context() -> IterationTraceContext | None:
    return _ITERATION_TRACE_CONTEXT.get()


@contextmanager
def iteration_trace_context(ctx: IterationTraceContext) -> Iterator[None]:
    token = _ITERATION_TRACE_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _ITERATION_TRACE_CONTEXT.reset(token)
