#!/usr/bin/env python3
"""Generate control/comm/compute evaluation plots for result folders."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow `python plotting_tools/evaluate_event_classification.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plotting_tools.classify import (  # noqa: E402
    CONTROL_FAMILY,
    EXCLUDE_CONTROL_PATTERNS,
    NETWORK_SUBS,
    classify_event,
    first_match_control_pattern,
)
from plotting_tools.comm_nvtx import (  # noqa: E402
    TRUSTED_COMM_PHASE_SOURCES,
    finalize_comm_nvtx_records,
    parse_comm_nvtx_label,
    parse_iter_nvtx_label,
)
from plotting_tools.nsys_jsonl import (  # noqa: E402
    _MEMCPY_KIND,
    _is_iteration_nvtx_name,
    _ns_to_us,
    _resolve_name,
    _resolve_nvtx_text,
)
from plotting_tools.trace_io import (  # noqa: E402
    parse_iteration_log,
    parse_job_metadata,
)

KIND_ORDER = ("compute", "comm", "control")
KIND_COLORS = {
    "compute": "#4e79a7",
    "comm": "#e15759",
    "control": "#59a14f",
}

SUB_ORDER = (
    "attention_comp",
    "matmul_gemm",
    "moe_expert",
    "moe_routing",
    "gate_comp",
    "add_norm_comp",
    "kv_cache_write",
    "rotary_embedding",
    "sampling_overhead",
    "masking_indexing",
    "other_compute",
    "network_collective",
    "network_p2p",
    "device_copy",
    "host_transfer",
    "control",
)

SUB_LABELS = {
    "attention_comp": "Attention",
    "matmul_gemm": "MatMul/GEMM",
    "moe_expert": "MoE Expert",
    "moe_routing": "MoE Routing",
    "gate_comp": "Gate",
    "add_norm_comp": "Norm/Add",
    "kv_cache_write": "KV Cache",
    "rotary_embedding": "Rotary",
    "sampling_overhead": "Sampling",
    "masking_indexing": "Mask/Index",
    "other_compute": "Other Compute",
    "network_collective": "Network Collective",
    "network_p2p": "Network P2P",
    "device_copy": "Device Copy",
    "host_transfer": "Host Transfer",
    "control": "Control",
}

CONTROL_FAMILY_ORDER = (
    "kernel_launch",
    "sync_poll",
    "runtime_meta",
    "memory_mgmt",
    "fabric_setup",
    "runtime_memory_op",
    "data_dependent",
    "framework_bookkeeping",
    "collective_setup",
    "unclassified_control",
)

SCHEDULER_PHASE_ORDER = ("prefill_only", "mixed", "decode_only", "idle")
SCHEDULER_PHASE_LABELS = {
    "prefill_only": "prefill only",
    "mixed": "mixed",
    "decode_only": "decode only",
    "idle": "idle",
}
SCHEDULER_PHASE_COLORS = {
    "prefill_only": "#f28e2b",
    "mixed": "#b07aa1",
    "decode_only": "#59a14f",
    "idle": "#9c9c9c",
}
TOKEN_SPLIT_ORDER = ("prefill_tokens", "decode_tokens")
TOKEN_SPLIT_LABELS = {
    "prefill_tokens": "prefill/context tokens",
    "decode_tokens": "decode/generation tokens",
}
TOKEN_SPLIT_COLORS = {
    "prefill_tokens": "#f28e2b",
    "decode_tokens": "#59a14f",
}

EVENT_PHASE_ORDER = ("prefill", "decode", "unknown")
RAW_EVENT_PHASE_ORDER = ("prefill", "decode", "mixed", "unknown")
EVENT_PHASE_COLORS = {
    "prefill": "#f28e2b",
    "decode": "#59a14f",
    "mixed": "#b07aa1",
    "unknown": "#9c9c9c",
}
KERNEL_PHASE_BUCKET_ORDER = ("prefill", "decode", "unknown")
KERNEL_PHASE_BUCKET_COLORS = {
    "prefill": "#f28e2b",
    "decode": "#59a14f",
    "unknown": "#9c9c9c",
}
MANUAL_PHASE_HEURISTIC_MIN_EVENTS = 16
MANUAL_PHASE_HEURISTIC_DOMINANCE = 0.90

HIST_BINS_US = np.logspace(-1, 7, 81)
PREFILL_COMM_CONTROL_WINDOW_US = 5_000
PREFILL_COMM_BURST_MERGE_GAP_US = 1_000
PREFILL_COMM_CONTROL_BIN_WIDTH_US = 100
PREFILL_COMM_CONTROL_BIN_EDGES_US = np.arange(
    -PREFILL_COMM_CONTROL_WINDOW_US,
    PREFILL_COMM_CONTROL_WINDOW_US + PREFILL_COMM_CONTROL_BIN_WIDTH_US,
    PREFILL_COMM_CONTROL_BIN_WIDTH_US,
)
FLASH_KERNEL_VARIANT_ORDER = (
    "run_flash_fwd",
    "run_flash_fwd_combine",
    "flash_fwd_other",
)
FLASH_KERNEL_LABELS = {
    "run_flash_fwd": "run_flash_fwd",
    "run_flash_fwd_combine": "run_flash_fwd_combine",
    "flash_fwd_other": "flash_fwd_other",
}
FLASH_KERNEL_COLORS = {
    "run_flash_fwd": "#4e79a7",
    "run_flash_fwd_combine": "#e15759",
    "flash_fwd_other": "#59a14f",
}


def _blank_hist() -> dict[str, np.ndarray]:
    return {kind: np.zeros(len(HIST_BINS_US) - 1, dtype=np.int64)
            for kind in KIND_ORDER}


def _blank_phase_kind() -> dict[str, Counter]:
    return {phase: Counter() for phase in EVENT_PHASE_ORDER}


def _new_stats(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "jsonl_files": [],
        "nsys_rep_files": [],
        "lines": 0,
        "decoded_rows": 0,
        "events": 0,
        "skipped_rows": Counter(),
        "count_by_kind": Counter(),
        "duration_us_by_kind": Counter(),
        "count_by_subcategory": Counter(),
        "duration_us_by_subcategory": Counter(),
        "count_by_control_family": Counter(),
        "duration_us_by_control_family": Counter(),
        "count_by_event": Counter(),
        "duration_us_by_event": Counter(),
        "unclassified": Counter(),
        "hist_by_kind": _blank_hist(),
        "raw_event_phase_counts": Counter(),
        "phase_count_by_kind": _blank_phase_kind(),
        "phase_duration_us_by_kind": _blank_phase_kind(),
        "jsonl_phase_ranges": Counter(),
        "jsonl_comm_phase_stats": Counter(),
        "heuristic_raw_event_phase_counts": Counter(),
        "heuristic_phase_count_by_kind": _blank_phase_kind(),
        "heuristic_phase_duration_us_by_kind": _blank_phase_kind(),
        "heuristic_trace_windows": [],
        "control_comm_burst_cluster": Counter(),
        "control_comm_burst_density_hist": np.zeros(
            len(PREFILL_COMM_CONTROL_BIN_EDGES_US) - 1,
            dtype=np.float64,
        ),
        "control_comm_burst_workers": [],
        "flash_kernel_events": [],
        "flash_comm_burst_cluster": Counter(),
        "flash_comm_burst_density_hist_by_variant": {
            variant: np.zeros(
                len(PREFILL_COMM_CONTROL_BIN_EDGES_US) - 1,
                dtype=np.float64,
            )
            for variant in FLASH_KERNEL_VARIANT_ORDER
        },
        "flash_comm_burst_workers": [],
        "kernel_name_stats": {},
        "workers": {},
        "iteration_logs": [],
        "scheduler_iterations": 0,
        "scheduler_phase_counts": Counter(),
        "scheduler_phase_elapsed_ms": Counter(),
        "scheduler_token_totals": Counter(),
        "scheduler_context_tokens_by_phase": Counter(),
        "scheduler_generation_tokens_by_phase": Counter(),
        "scheduler_token_summary": {},
    }


def _worker_stats(stats: dict[str, Any], worker: str) -> dict[str, Any]:
    workers = stats["workers"]
    if worker not in workers:
        workers[worker] = {
            "count_by_kind": Counter(),
            "duration_us_by_kind": Counter(),
            "count_by_subcategory": Counter(),
            "duration_us_by_subcategory": Counter(),
            "count_by_control_family": Counter(),
            "duration_us_by_control_family": Counter(),
            "unclassified": Counter(),
            "hist_by_kind": _blank_hist(),
            "events": 0,
        }
    return workers[worker]


def _experiment_label(path: Path) -> str:
    parts: dict[str, str] = {}
    for token in path.name.split("_"):
        if token.startswith("sp") and token[2:].isdigit():
            parts["SP"] = token[2:]
        elif token.startswith("sd") and token[2:].isdigit():
            parts["SD"] = token[2:]
        elif token.startswith("tp") and token[2:].isdigit():
            parts["TP"] = token[2:]
    return " ".join(f"{k}{v}" for k, v in parts.items()) or path.name


def _worker_label(path: Path) -> str:
    stem = path.stem
    marker = "worker_process_"
    if marker in stem:
        return "worker_" + stem.rsplit(marker, 1)[1]
    return stem


def _hist_add(hist: dict[str, np.ndarray], kind: str, dur_us: int) -> None:
    if kind not in hist:
        return
    value = max(float(dur_us), HIST_BINS_US[0])
    idx = int(np.searchsorted(HIST_BINS_US, value, side="right") - 1)
    idx = max(0, min(idx, len(HIST_BINS_US) - 2))
    hist[kind][idx] += 1


def _flash_kernel_variant(name: str) -> str | None:
    lower = name.lower()
    compact = lower.replace("_", "")
    if (
        "run_flash_fwd_combine" in lower
        or "flash_fwd_splitkv_mla_combine_kernel" in lower
        or "flashattnfwdcombine" in compact
    ):
        return "run_flash_fwd_combine"
    if (
        "run_flash_fwd" in lower
        or "flash_fwd_splitkv_mla_kernel" in lower
        or "flash_fwd_kernel" in lower
        or "flashattnfwd" in compact
    ):
        return "run_flash_fwd"
    if "flash_fwd" in lower:
        return "flash_fwd_other"
    return None


def _kernel_name_stats(stats: dict[str, Any], name: str) -> dict[str, Any]:
    kernel_stats = stats["kernel_name_stats"]
    if name not in kernel_stats:
        kernel_stats[name] = {
            "count": 0,
            "duration_us": 0,
            "durations_us": [],
            "heuristic_phase_counts": Counter(),
            "kind_counts": Counter(),
            "subcategory_counts": Counter(),
        }
    return kernel_stats[name]


def _record_kernel_name_event(
    stats: dict[str, Any],
    *,
    name: str,
    dur_us: int,
    kind: str,
    subcategory: str,
    heuristic_phase: str | None,
) -> None:
    row = _kernel_name_stats(stats, name)
    row["count"] += 1
    row["duration_us"] += dur_us
    row["durations_us"].append(dur_us)
    phase = heuristic_phase if heuristic_phase in RAW_EVENT_PHASE_ORDER else "unknown"
    row["heuristic_phase_counts"][phase] += 1
    row["kind_counts"][kind] += 1
    row["subcategory_counts"][subcategory] += 1


def _kernel_name_phase_rule(name: str) -> tuple[str | None, str]:
    lower = name.lower()
    compact = lower.replace("_", "")

    decode_patterns = (
        "decode",
        "paged_attention",
        "pagedattention",
        "splitkv",
        "flash_fwd_splitkv",
        "flashattnfwdcombine",
        "apply_penalty",
        "sampling",
        "sample",
        "top_p",
        "topp",
        "mask_top_p",
        "multinomial",
        "argmax",
        "rejection",
        "revert_output_bin_count",
    )
    for pattern in decode_patterns:
        if pattern in lower or pattern in compact:
            return "decode", f"name pattern: {pattern}"

    prefill_patterns = (
        "prefill",
        "prompt",
        "context_attention",
        "contextattention",
        "context_fwd",
    )
    for pattern in prefill_patterns:
        if pattern in lower or pattern in compact:
            return "prefill", f"name pattern: {pattern}"

    return None, "no strong name pattern"


def _is_attention_like_kernel_name(name: str) -> bool:
    lower = name.lower()
    compact = lower.replace("_", "")
    return any(
        pattern in lower or pattern in compact
        for pattern in (
            "flash_fwd",
            "flashattnfwd",
            "paged_attention",
            "pagedattention",
            "attention",
            "mla_kernel",
        )
    )


def _kernel_observed_phase_rule(
    name: str,
    *,
    count: int,
    p50_ms: float,
    p95_ms: float,
) -> tuple[str | None, str]:
    if not _is_attention_like_kernel_name(name):
        return None, "not attention-like"

    if count >= 100 and p95_ms <= 0.05:
        return "decode", "short attention duration shape"
    if p50_ms >= 0.05 or p95_ms >= 0.20:
        return "prefill", "long attention duration shape"
    return None, "attention duration shape ambiguous"


def _short_kernel_name(name: str, *, max_len: int = 96) -> str:
    compact = name.replace("_", "")
    for marker in (
        "FlashAttnFwdCombine",
        "FlashAttnFwdSm90",
        "reshape_and_cache_flash_kernel",
        "prepare_varlen_num_blocks_kernel",
        "ncclDevKernel",
        "paged_attention",
    ):
        if marker in name or marker.replace("_", "") in compact:
            return marker
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def _control_family(name: str, cat: str) -> str:
    s = f"{name} {cat}".lower()
    pattern = first_match_control_pattern(s)
    if pattern is None:
        return "unclassified_control"
    return CONTROL_FAMILY.get(pattern, "unclassified_control")


def _credit_phase(
    raw_counts: Counter,
    count_by_kind: dict[str, Counter],
    duration_by_kind: dict[str, Counter],
    *,
    phase: str,
    kind: str,
    dur_us: int,
) -> None:
    phase = phase if phase in RAW_EVENT_PHASE_ORDER else "unknown"
    raw_counts[phase] += 1
    credited_phases = ("prefill", "decode") if phase == "mixed" else (phase,)
    for credited_phase in credited_phases:
        if credited_phase not in EVENT_PHASE_ORDER:
            credited_phase = "unknown"
        count_by_kind[credited_phase][kind] += 1
        duration_by_kind[credited_phase][kind] += dur_us


def _record_event(
    stats: dict[str, Any],
    worker_stats: dict[str, Any],
    *,
    name: str,
    cat: str,
    dur_us: int,
    args: dict[str, Any],
    phase: str = "unknown",
    heuristic_phase: str | None = None,
) -> tuple[str, str]:
    unclassified: list[str] = []
    kind, sub = classify_event(name, cat, unclassified, args=args)
    key = f"{cat}|{name}"

    stats["events"] += 1
    worker_stats["events"] += 1
    stats["count_by_kind"][kind] += 1
    worker_stats["count_by_kind"][kind] += 1
    stats["duration_us_by_kind"][kind] += dur_us
    worker_stats["duration_us_by_kind"][kind] += dur_us
    stats["count_by_subcategory"][sub] += 1
    worker_stats["count_by_subcategory"][sub] += 1
    stats["duration_us_by_subcategory"][sub] += dur_us
    worker_stats["duration_us_by_subcategory"][sub] += dur_us
    stats["count_by_event"][key] += 1
    stats["duration_us_by_event"][key] += dur_us
    _hist_add(stats["hist_by_kind"], kind, dur_us)
    _hist_add(worker_stats["hist_by_kind"], kind, dur_us)
    if cat == "kernel":
        _record_kernel_name_event(
            stats,
            name=name,
            dur_us=dur_us,
            kind=kind,
            subcategory=sub,
            heuristic_phase=heuristic_phase,
        )

    _credit_phase(
        stats["raw_event_phase_counts"],
        stats["phase_count_by_kind"],
        stats["phase_duration_us_by_kind"],
        phase=phase,
        kind=kind,
        dur_us=dur_us,
    )
    if heuristic_phase is not None:
        _credit_phase(
            stats["heuristic_raw_event_phase_counts"],
            stats["heuristic_phase_count_by_kind"],
            stats["heuristic_phase_duration_us_by_kind"],
            phase=heuristic_phase,
            kind=kind,
            dur_us=dur_us,
        )

    if kind == "control":
        family = _control_family(name, cat)
        stats["count_by_control_family"][family] += 1
        worker_stats["count_by_control_family"][family] += 1
        stats["duration_us_by_control_family"][family] += dur_us
        worker_stats["duration_us_by_control_family"][family] += dur_us

    for item in unclassified:
        stats["unclassified"][item] += 1
        worker_stats["unclassified"][item] += 1

    return kind, sub


def _trusted_comm_phase(record: dict[str, Any]) -> str | None:
    phase = (record.get("phase") or "unknown").lower().strip()
    if phase not in ("prefill", "decode", "mixed"):
        return None
    source = (record.get("phase_source") or "comm_nvtx_label").strip()
    if source not in TRUSTED_COMM_PHASE_SOURCES:
        return None
    return phase


def _event_phase_from_scheduler_phase(phase: str) -> str:
    phase = phase.lower().strip()
    if phase == "prefill_only":
        return "prefill"
    if phase == "decode_only":
        return "decode"
    if phase == "mixed":
        return "mixed"
    return "unknown"


def _update_trace_window(
    row: dict[str, Any],
    trace_window: dict[str, int | None],
) -> None:
    start = row.get("start")
    end = row.get("end")
    if start is None or end is None:
        return
    dur_ns = int(end) - int(start)
    if dur_ns <= 0:
        return
    start_us = _ns_to_us(int(start))
    end_us = _ns_to_us(int(end))
    current_min = trace_window.get("start_us")
    current_max = trace_window.get("end_us")
    if current_min is None or start_us < int(current_min):
        trace_window["start_us"] = start_us
    if current_max is None or end_us > int(current_max):
        trace_window["end_us"] = end_us


def collect_jsonl_phase_ranges(
    path: Path,
    stats: dict[str, Any],
) -> tuple[list[tuple[str, int, int, int, str]], dict[str, int | None]]:
    strings: dict[int, str] = {}
    iteration_ranges: list[dict[str, Any]] = []
    comm_records: list[dict[str, Any]] = []
    trace_window: dict[str, int | None] = {"start_us": None, "end_us": None}

    with path.open() as f:
        for line in f:
            if '"table":"StringIds"' in line:
                row = json.loads(line)
                strings[int(row["id"])] = row["value"]
                continue
            if "CUPTI_ACTIVITY_KIND" in line:
                row = json.loads(line)
                if str(row.get("table", "")).startswith("CUPTI_ACTIVITY_KIND"):
                    _update_trace_window(row, trace_window)
                continue
            if "NVTX_EVENTS" not in line:
                continue

            row = json.loads(line)
            if row.get("table") != "NVTX_EVENTS":
                continue
            start = row.get("start")
            end = row.get("end")
            if start is None or end is None:
                continue
            dur_ns = int(end) - int(start)
            if dur_ns <= 0:
                continue

            name = _resolve_nvtx_text(row, strings)
            ts_us = _ns_to_us(int(start))
            end_us = _ns_to_us(int(end))
            if _is_iteration_nvtx_name(name):
                parsed_iter = parse_iter_nvtx_label(name)
                if parsed_iter is not None:
                    iteration_ranges.append({
                        **parsed_iter,
                        "ts": ts_us,
                        "end": end_us,
                    })
                else:
                    phase = name.lower().strip()
                    iteration_ranges.append({
                        "name": phase,
                        "scope": "iteration",
                        "phase": phase,
                        "ts": ts_us,
                        "end": end_us,
                    })
            elif parsed_comm := parse_comm_nvtx_label(name):
                comm_records.append({
                    **parsed_comm,
                    "ts": ts_us,
                    "end": end_us,
                    "dur_us": end_us - ts_us,
                    "scope": "comm",
                })

    phase_ranges: list[tuple[str, int, int, int, str]] = []
    for row in iteration_ranges:
        phase = (row.get("phase") or row.get("name") or "").lower().strip()
        if phase not in ("prefill", "decode", "mixed"):
            continue
        start = int(row["ts"])
        end = int(row["end"])
        phase_ranges.append((phase, start, end, end - start, "iteration_nvtx"))
        stats["jsonl_phase_ranges"][f"iteration:{phase}"] += 1

    if comm_records:
        comm_records, comm_stats = finalize_comm_nvtx_records(
            comm_records,
            iteration_ranges,
        )
        for key, value in comm_stats.items():
            stats["jsonl_comm_phase_stats"][key] += int(value)
        for record in comm_records:
            phase = _trusted_comm_phase(record)
            if phase is None:
                continue
            start = int(record["ts"])
            end = int(record["end"])
            phase_ranges.append((phase, start, end, end - start, "comm_nvtx"))
            stats["jsonl_phase_ranges"][f"comm:{phase}"] += 1

    return sorted(phase_ranges, key=lambda item: item[1]), trace_window


class SchedulerPhaseWarper:
    def __init__(
        self,
        iterations: list[dict[str, Any]],
        trace_start_us: int | None,
        trace_end_us: int | None,
    ) -> None:
        self.available = False
        self.starts_us: list[int] = []
        self.ends_us: list[int] = []
        self.phases: list[str] = []
        self.trace_start_us = trace_start_us
        self.trace_end_us = trace_end_us
        self.trace_span_us = 0
        self.scheduler_span_us = 0
        if trace_start_us is None or trace_end_us is None:
            return
        self.trace_span_us = int(trace_end_us) - int(trace_start_us)
        if self.trace_span_us <= 0 or not iterations:
            return

        cursor_us = 0
        for iteration in iterations:
            elapsed_us = int(round(float(iteration.get("elapsed_ms", 0.0)) * 1000.0))
            if elapsed_us <= 0:
                continue
            self.starts_us.append(cursor_us)
            cursor_us += elapsed_us
            self.ends_us.append(cursor_us)
            self.phases.append(
                _event_phase_from_scheduler_phase(str(iteration.get("phase", "")))
            )

        self.scheduler_span_us = cursor_us
        self.available = self.scheduler_span_us > 0 and bool(self.starts_us)

    def phase_for_event(self, ts: int, end: int) -> str:
        if not self.available or self.trace_start_us is None:
            return "unknown"
        midpoint_us = (ts + end) // 2
        rel = (midpoint_us - int(self.trace_start_us)) / self.trace_span_us
        rel = max(0.0, min(1.0, rel))
        scheduler_us = int(round(rel * self.scheduler_span_us))
        idx = bisect_right(self.starts_us, scheduler_us) - 1
        if idx < 0:
            return "unknown"
        if scheduler_us >= self.ends_us[idx]:
            return "unknown"
        return self.phases[idx]

    def summary(self, worker: str) -> dict[str, Any]:
        return {
            "worker": worker,
            "available": self.available,
            "trace_start_us": self.trace_start_us,
            "trace_end_us": self.trace_end_us,
            "trace_span_seconds": round(self.trace_span_us / 1e6, 6),
            "scheduler_span_seconds": round(self.scheduler_span_us / 1e6, 6),
            "mapping": "linear time-warp from CUPTI trace span to scheduler elapsed",
        }


class PhaseRangeIndex:
    def __init__(self, ranges: list[tuple[str, int, int, int, str]]) -> None:
        self.ranges = ranges
        self.bin_us = 10_000
        self.bins: dict[int, list[int]] = defaultdict(list)
        for idx, (_phase, start, end, _span, _source) in enumerate(ranges):
            first_bin = start // self.bin_us
            last_bin = max(start, end - 1) // self.bin_us
            for bin_idx in range(first_bin, last_bin + 1):
                self.bins[bin_idx].append(idx)

    def phase_for_event(self, ts: int, end: int) -> str:
        if not self.ranges:
            return "unknown"

        first_bin = ts // self.bin_us
        last_bin = max(ts, end - 1) // self.bin_us
        seen: set[int] = set()
        best_phase = "unknown"
        best_span: int | None = None
        for bin_idx in range(first_bin, last_bin + 1):
            for range_idx in self.bins.get(bin_idx, ()):
                if range_idx in seen:
                    continue
                seen.add(range_idx)
                phase, start, range_end, span, _source = self.ranges[range_idx]
                if ts < range_end and end > start:
                    if best_span is None or span < best_span:
                        best_phase = phase
                        best_span = span
        return best_phase


def _merge_comm_bursts(
    bursts: list[tuple[int, int]],
    gap_us: int = PREFILL_COMM_BURST_MERGE_GAP_US,
) -> list[tuple[int, int]]:
    if not bursts:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(bursts):
        start = int(start)
        end = max(int(end), start + 1)
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_us:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _record_control_comm_burst_clustering(
    stats: dict[str, Any],
    worker: str,
    control_starts_us: list[int],
    comm_bursts_us: list[tuple[int, int]],
    fabric_comm_events: int,
) -> None:
    cluster = stats["control_comm_burst_cluster"]
    merged_bursts = _merge_comm_bursts(comm_bursts_us)
    cluster["control_events"] += len(control_starts_us)
    cluster["comm_events"] += len(comm_bursts_us)
    cluster["fabric_comm_events"] += fabric_comm_events
    cluster["raw_comm_bursts"] += len(comm_bursts_us)
    cluster["merged_comm_bursts"] += len(merged_bursts)

    worker_summary: dict[str, Any] = {
        "worker": worker,
        "control_events": len(control_starts_us),
        "comm_events": len(comm_bursts_us),
        "fabric_comm_events": fabric_comm_events,
        "raw_comm_bursts": len(comm_bursts_us),
        "merged_comm_bursts": len(merged_bursts),
        "comm_bursts": 0,
        "clustered_control_events": 0,
        "clustered_control_fraction": 0.0,
        "pre_burst_control_sum": 0,
        "during_burst_control_sum": 0,
        "post_burst_control_sum": 0,
        "burst_duration_us_sum": 0,
    }
    stats["control_comm_burst_workers"].append(worker_summary)

    if not control_starts_us or not merged_bursts:
        return

    controls = np.array(sorted(control_starts_us), dtype=np.int64)
    comm_starts = sorted(start for start, _end in merged_bursts)

    clustered_controls = 0
    for control_start in controls:
        idx = bisect_left(comm_starts, control_start)
        best_distance = None
        if idx < len(comm_starts):
            best_distance = abs(int(control_start) - comm_starts[idx])
        if idx > 0:
            left_distance = abs(int(control_start) - comm_starts[idx - 1])
            if best_distance is None or left_distance < best_distance:
                best_distance = left_distance
        if (
            best_distance is not None
            and best_distance <= PREFILL_COMM_CONTROL_WINDOW_US
        ):
            clustered_controls += 1

    hist = np.zeros(len(PREFILL_COMM_CONTROL_BIN_EDGES_US) - 1, dtype=np.float64)
    pre_burst_sum = 0
    during_burst_sum = 0
    post_burst_sum = 0
    burst_duration_sum = 0
    for comm_start, comm_end in merged_bursts:
        burst_duration = max(int(comm_end) - int(comm_start), 1)
        burst_duration_sum += burst_duration
        rel = controls - int(comm_start)
        in_window = (rel >= -PREFILL_COMM_CONTROL_WINDOW_US) & (
            rel <= PREFILL_COMM_CONTROL_WINDOW_US
        )
        if np.any(in_window):
            hist += np.histogram(
                rel[in_window],
                bins=PREFILL_COMM_CONTROL_BIN_EDGES_US,
            )[0]
        pre_burst_sum += int(
            np.sum((rel < 0) & (rel >= -PREFILL_COMM_CONTROL_WINDOW_US))
        )
        during_burst_sum += int(np.sum((rel >= 0) & (rel <= burst_duration)))
        post_burst_sum += int(
            np.sum(
                (rel > burst_duration)
                & (rel <= PREFILL_COMM_CONTROL_WINDOW_US)
            )
        )

    stats["control_comm_burst_density_hist"] += hist
    cluster["comm_bursts"] += len(merged_bursts)
    cluster["clustered_control_events"] += clustered_controls
    cluster["pre_burst_control_sum"] += pre_burst_sum
    cluster["during_burst_control_sum"] += during_burst_sum
    cluster["post_burst_control_sum"] += post_burst_sum
    cluster["burst_duration_us_sum"] += burst_duration_sum

    worker_summary["comm_bursts"] = len(merged_bursts)
    worker_summary["clustered_control_events"] = clustered_controls
    worker_summary["clustered_control_fraction"] = round(
        clustered_controls / len(controls),
        6,
    )
    worker_summary["pre_burst_control_sum"] = pre_burst_sum
    worker_summary["during_burst_control_sum"] = during_burst_sum
    worker_summary["post_burst_control_sum"] = post_burst_sum
    worker_summary["burst_duration_us_sum"] = burst_duration_sum


def _record_flash_comm_burst_pattern(
    stats: dict[str, Any],
    worker: str,
    flash_events: list[dict[str, Any]],
    comm_bursts_us: list[tuple[int, int]],
) -> None:
    cluster = stats["flash_comm_burst_cluster"]
    merged_bursts = _merge_comm_bursts(comm_bursts_us)
    cluster["flash_events"] += len(flash_events)
    cluster["comm_bursts"] += len(merged_bursts)

    counts_by_variant = Counter(event["variant"] for event in flash_events)
    for variant, count in counts_by_variant.items():
        cluster[f"events:{variant}"] += int(count)

    worker_summary = {
        "worker": worker,
        "flash_events": len(flash_events),
        "comm_bursts": len(merged_bursts),
        "counts_by_variant": dict(counts_by_variant),
    }
    stats["flash_comm_burst_workers"].append(worker_summary)

    if not flash_events or not merged_bursts:
        return

    starts_by_variant: dict[str, np.ndarray] = {}
    for variant in FLASH_KERNEL_VARIANT_ORDER:
        starts = sorted(
            int(event["start_us"])
            for event in flash_events
            if event["variant"] == variant
        )
        starts_by_variant[variant] = np.array(starts, dtype=np.int64)

    for comm_start, _comm_end in merged_bursts:
        for variant, starts in starts_by_variant.items():
            if starts.size == 0:
                continue
            rel = starts - int(comm_start)
            in_window = (rel >= -PREFILL_COMM_CONTROL_WINDOW_US) & (
                rel <= PREFILL_COMM_CONTROL_WINDOW_US
            )
            if not np.any(in_window):
                continue
            stats["flash_comm_burst_density_hist_by_variant"][variant] += (
                np.histogram(
                    rel[in_window],
                    bins=PREFILL_COMM_CONTROL_BIN_EDGES_US,
                )[0]
            )


def scan_jsonl(
    path: Path,
    stats: dict[str, Any],
    scheduler_iterations: list[dict[str, Any]],
) -> None:
    phase_ranges, trace_window = collect_jsonl_phase_ranges(path, stats)
    phase_index = PhaseRangeIndex(phase_ranges)
    strings: dict[int, str] = {}
    worker = _worker_label(path)
    heuristic_index = SchedulerPhaseWarper(
        scheduler_iterations,
        trace_window["start_us"],
        trace_window["end_us"],
    )
    stats["heuristic_trace_windows"].append(heuristic_index.summary(worker))
    worker_stats = _worker_stats(stats, worker)
    control_starts_us: list[int] = []
    comm_bursts_us: list[tuple[int, int]] = []
    fabric_comm_events = 0
    flash_events: list[dict[str, Any]] = []
    print(
        f"  scanning {path} ({len(phase_ranges):,} phase ranges, "
        f"heuristic={heuristic_index.available})",
        flush=True,
    )

    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if line_no % 5_000_000 == 0:
                print(
                    f"    {path.name}: {line_no:,} lines, "
                    f"{stats['events']:,} events",
                    flush=True,
                )

            if '"table":"StringIds"' in line:
                row = json.loads(line)
                strings[int(row["id"])] = row["value"]
                stats["decoded_rows"] += 1
                continue
            if "CUPTI_ACTIVITY_KIND" not in line:
                continue

            row = json.loads(line)
            stats["decoded_rows"] += 1
            table = row.get("table")
            start = row.get("start")
            end = row.get("end")
            if start is None or end is None:
                stats["skipped_rows"]["no_start_end"] += 1
                continue
            dur_ns = int(end) - int(start)
            if dur_ns <= 0:
                stats["skipped_rows"]["zero_duration"] += 1
                continue
            ts = _ns_to_us(int(start))
            dur_us = dur_ns // 1000

            args: dict[str, Any] = {}
            if row.get("deviceId") is not None:
                args["device_id"] = int(row["deviceId"])
            if row.get("streamId") is not None:
                args["stream_id"] = int(row["streamId"])

            if table == "CUPTI_ACTIVITY_KIND_KERNEL":
                name = _resolve_name(row, strings)
                cat = "kernel"
            elif table == "CUPTI_ACTIVITY_KIND_MEMCPY":
                copy_kind = int(row.get("copyKind", 0))
                name = _MEMCPY_KIND.get(copy_kind, "memcpy")
                cat = "memcpy"
                args["copy_kind"] = copy_kind
                if row.get("bytes") is not None:
                    args["bytes"] = row["bytes"]
            elif table == "CUPTI_ACTIVITY_KIND_RUNTIME":
                name = strings.get(int(row["nameId"]), "")
                lower = name.lower()
                if any(pattern in lower for pattern in EXCLUDE_CONTROL_PATTERNS):
                    stats["skipped_rows"]["profiler"] += 1
                    continue
                cat = "runtime"
            else:
                stats["skipped_rows"][f"table:{table}"] += 1
                continue

            event_end = ts + max(dur_us, 1)
            exact_phase = phase_index.phase_for_event(ts, event_end)
            heuristic_phase = heuristic_index.phase_for_event(ts, event_end)
            kind, sub = _record_event(
                stats,
                worker_stats,
                name=name,
                cat=cat,
                dur_us=dur_us,
                args=args,
                phase=exact_phase,
                heuristic_phase=heuristic_phase,
            )
            if kind == "control":
                control_starts_us.append(ts)
            elif kind == "comm":
                comm_bursts_us.append((ts, event_end))
                if sub in NETWORK_SUBS:
                    fabric_comm_events += 1

            if cat == "kernel":
                flash_variant = _flash_kernel_variant(name)
                if flash_variant is not None:
                    event = {
                        "worker": worker,
                        "start_us": ts,
                        "dur_us": dur_us,
                        "variant": flash_variant,
                        "heuristic_phase": heuristic_phase,
                    }
                    stats["flash_kernel_events"].append(event)
                    flash_events.append(event)

    stats["lines"] += line_no if "line_no" in locals() else 0
    _record_control_comm_burst_clustering(
        stats,
        worker,
        control_starts_us,
        comm_bursts_us,
        fabric_comm_events,
    )
    _record_flash_comm_burst_pattern(
        stats,
        worker,
        flash_events,
        comm_bursts_us,
    )


def _scheduler_phase(iteration: dict[str, Any]) -> str:
    context_tokens = int(iteration.get("context_tokens", 0))
    generation_tokens = int(iteration.get("generation_tokens", 0))
    if context_tokens > 0 and generation_tokens > 0:
        return "mixed"
    if context_tokens > 0:
        return "prefill_only"
    if generation_tokens > 0:
        return "decode_only"
    return "idle"


def _percentile(values: np.ndarray, pct: float) -> float:
    return float(np.percentile(values, pct)) if values.size else 0.0


def _record_scheduler_iterations(
    stats: dict[str, Any],
    iterations: list[dict[str, Any]],
    *,
    max_num_batched_tokens: int | None,
) -> None:
    stats["scheduler_iterations"] = len(iterations)
    if not iterations:
        return

    phase_counts: Counter = Counter()
    phase_elapsed: Counter = Counter()
    context_by_phase: Counter = Counter()
    generation_by_phase: Counter = Counter()

    context_tokens: list[int] = []
    generation_tokens: list[int] = []
    total_tokens: list[int] = []
    elapsed_ms: list[float] = []

    for iteration in iterations:
        phase = _scheduler_phase(iteration)
        ctx = int(iteration.get("context_tokens", 0))
        gen = int(iteration.get("generation_tokens", 0))
        elapsed = float(iteration.get("elapsed_ms", 0.0))

        phase_counts[phase] += 1
        phase_elapsed[phase] += elapsed
        context_by_phase[phase] += ctx
        generation_by_phase[phase] += gen
        context_tokens.append(ctx)
        generation_tokens.append(gen)
        total_tokens.append(ctx + gen)
        elapsed_ms.append(elapsed)

    total_arr = np.array(total_tokens, dtype=np.float64)
    positive_total = total_arr[total_arr > 0]
    elapsed_arr = np.array(elapsed_ms, dtype=np.float64)
    context_total = int(sum(context_tokens))
    generation_total = int(sum(generation_tokens))
    token_total = context_total + generation_total

    stats["scheduler_phase_counts"] = phase_counts
    stats["scheduler_phase_elapsed_ms"] = phase_elapsed
    stats["scheduler_context_tokens_by_phase"] = context_by_phase
    stats["scheduler_generation_tokens_by_phase"] = generation_by_phase
    stats["scheduler_token_totals"] = Counter({
        "prefill_tokens": context_total,
        "decode_tokens": generation_total,
    })
    stats["scheduler_token_summary"] = {
        "iterations": len(iterations),
        "max_num_batched_tokens": max_num_batched_tokens,
        "prefill_tokens_total": context_total,
        "decode_tokens_total": generation_total,
        "prefill_token_fraction": (
            context_total / token_total if token_total else 0.0
        ),
        "decode_token_fraction": (
            generation_total / token_total if token_total else 0.0
        ),
        "prefill_active_iterations": sum(1 for value in context_tokens if value > 0),
        "decode_active_iterations": sum(
            1 for value in generation_tokens if value > 0
        ),
        "prefill_only_iterations": int(phase_counts["prefill_only"]),
        "decode_only_iterations": int(phase_counts["decode_only"]),
        "mixed_prefill_decode_iterations": int(phase_counts["mixed"]),
        "idle_iterations": int(phase_counts["idle"]),
        "elapsed_seconds_total": float(elapsed_arr.sum() / 1000.0),
        "tokens_per_iteration": {
            "median_positive": _percentile(positive_total, 50),
            "p95": _percentile(total_arr, 95),
            "max": float(total_arr.max()) if total_arr.size else 0.0,
        },
    }


def collect_scheduler_iterations(
    path: Path,
    stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], int | None]:
    iterations: list[dict[str, Any]] = []
    max_num_batched_tokens: int | None = None

    for slurm_out in sorted(path.glob("*.out")):
        parsed = parse_iteration_log(slurm_out)
        if not parsed:
            continue
        stats["iteration_logs"].append(str(slurm_out))
        meta = parse_job_metadata(slurm_out)
        if max_num_batched_tokens is None:
            raw_max = meta.get("max_num_batched_tokens")
            if raw_max is not None:
                max_num_batched_tokens = int(raw_max)
        for iteration in parsed:
            phase = _scheduler_phase(iteration)
            ctx = int(iteration.get("context_tokens", 0))
            gen = int(iteration.get("generation_tokens", 0))
            iterations.append({
                **iteration,
                "phase": phase,
                "prefill_tokens": ctx,
                "decode_tokens": gen,
                "total_tokens": ctx + gen,
                "source_log": str(slurm_out),
            })

    _record_scheduler_iterations(
        stats,
        iterations,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    return iterations, max_num_batched_tokens


def _counter_json(counter: Counter, *, scale: float = 1.0) -> dict[str, float]:
    return {
        key: round(float(value) / scale, 6)
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def _nested_counter_json(
    counters: dict[str, Counter],
    *,
    scale: float = 1.0,
) -> dict[str, dict[str, float]]:
    return {
        outer_key: _counter_json(counter, scale=scale)
        for outer_key, counter in counters.items()
    }


def _hist_percentile_ms(counts: np.ndarray, pct: float) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0

    threshold = total * pct / 100.0
    cumulative = 0
    for idx, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        next_cumulative = cumulative + count
        if next_cumulative >= threshold:
            fraction = (threshold - cumulative) / count
            lo = PREFILL_COMM_CONTROL_BIN_EDGES_US[idx] / 1000.0
            hi = PREFILL_COMM_CONTROL_BIN_EDGES_US[idx + 1] / 1000.0
            return round(float(lo + fraction * (hi - lo)), 6)
        cumulative = next_cumulative
    return round(float(PREFILL_COMM_CONTROL_BIN_EDGES_US[-1] / 1000.0), 6)


def _control_comm_burst_cluster_json(stats: dict[str, Any]) -> dict[str, Any]:
    cluster = stats["control_comm_burst_cluster"]
    control_events = int(cluster.get("control_events", 0))
    clustered_controls = int(cluster.get("clustered_control_events", 0))
    hist_sum = stats["control_comm_burst_density_hist"]
    comm_bursts = int(cluster.get("comm_bursts", 0))
    hist = (
        hist_sum / comm_bursts
        if comm_bursts > 0
        else np.zeros(len(PREFILL_COMM_CONTROL_BIN_EDGES_US) - 1, dtype=np.float64)
    )
    mean_burst_duration_ms = (
        float(cluster.get("burst_duration_us_sum", 0)) / comm_bursts / 1000.0
        if comm_bursts > 0
        else 0.0
    )

    return {
        "note": (
            "Phase-agnostic: all classified control and comm events in the "
            "trace window. Communication anchors are comm event starts "
            "(burst=0); fabric_comm_events reports the "
            "network_collective/network_p2p subset."
        ),
        "phase_filter": "none (all phases)",
        "window_ms": round(PREFILL_COMM_CONTROL_WINDOW_US / 1000.0, 6),
        "time_axis_definition": (
            "control_event_start_ms - merged_comm_burst_start_ms; 0 is burst start"
        ),
        "comm_burst_merge_gap_ms": round(
            PREFILL_COMM_BURST_MERGE_GAP_US / 1000.0,
            6,
        ),
        "control_events": control_events,
        "comm_events": int(cluster.get("comm_events", 0)),
        "raw_comm_bursts": int(cluster.get("raw_comm_bursts", 0)),
        "comm_bursts": comm_bursts,
        "merged_comm_bursts": int(cluster.get("merged_comm_bursts", comm_bursts)),
        "fabric_comm_events": int(cluster.get("fabric_comm_events", 0)),
        "mean_burst_duration_ms": round(mean_burst_duration_ms, 6),
        "clustered_control_events": clustered_controls,
        "clustered_control_fraction": round(
            clustered_controls / control_events if control_events else 0.0,
            6,
        ),
        "mean_control_per_burst": {
            "pre_burst_window": round(
                float(cluster.get("pre_burst_control_sum", 0)) / comm_bursts,
                6,
            )
            if comm_bursts
            else 0.0,
            "during_burst": round(
                float(cluster.get("during_burst_control_sum", 0)) / comm_bursts,
                6,
            )
            if comm_bursts
            else 0.0,
            "post_burst_window": round(
                float(cluster.get("post_burst_control_sum", 0)) / comm_bursts,
                6,
            )
            if comm_bursts
            else 0.0,
        },
        "burst_relative_histogram_bin_edges_ms": [
            round(float(edge / 1000.0), 6)
            for edge in PREFILL_COMM_CONTROL_BIN_EDGES_US
        ],
        "burst_relative_histogram_mean_density": [
            round(float(value), 6) for value in hist.tolist()
        ],
        "workers": stats["control_comm_burst_workers"],
    }


def _flash_kernel_summary_json(stats: dict[str, Any]) -> dict[str, Any]:
    events = stats["flash_kernel_events"]
    cluster = stats["flash_comm_burst_cluster"]
    comm_bursts = int(cluster.get("comm_bursts", 0))

    counts_by_variant = Counter(event["variant"] for event in events)
    counts_by_phase = Counter(event["heuristic_phase"] for event in events)
    durations_by_variant: dict[str, dict[str, float]] = {}
    for variant in FLASH_KERNEL_VARIANT_ORDER:
        arr = np.array(
            [
                float(event["dur_us"]) / 1000.0
                for event in events
                if event["variant"] == variant
            ],
            dtype=np.float64,
        )
        durations_by_variant[variant] = {
            "count": int(arr.size),
            "p50_ms": round(_percentile(arr, 50), 6),
            "p95_ms": round(_percentile(arr, 95), 6),
            "max_ms": round(float(arr.max()) if arr.size else 0.0, 6),
        }

    density_by_variant: dict[str, list[float]] = {}
    for variant in FLASH_KERNEL_VARIANT_ORDER:
        hist = stats["flash_comm_burst_density_hist_by_variant"][variant]
        if comm_bursts > 0:
            hist = hist / comm_bursts
        density_by_variant[variant] = [
            round(float(value), 6) for value in hist.tolist()
        ]

    return {
        "note": (
            "Matched forward attention-like kernels by event name: "
            "run_flash_fwd, run_flash_fwd_combine, "
            "flash_fwd_splitkv_mla_kernel, and flash_fwd_kernel. "
            "reshape/cache and prepare_varlen kernels are excluded."
        ),
        "time_axis_definition": (
            "flash_kernel_start_ms - merged_comm_burst_start_ms; "
            "0 is communication burst start"
        ),
        "window_ms": round(PREFILL_COMM_CONTROL_WINDOW_US / 1000.0, 6),
        "comm_burst_merge_gap_ms": round(
            PREFILL_COMM_BURST_MERGE_GAP_US / 1000.0,
            6,
        ),
        "flash_events": len(events),
        "counts_by_variant": dict(counts_by_variant),
        "counts_by_heuristic_phase": dict(counts_by_phase),
        "duration_ms_by_variant": durations_by_variant,
        "comm_bursts": comm_bursts,
        "burst_relative_histogram_bin_edges_ms": [
            round(float(edge / 1000.0), 6)
            for edge in PREFILL_COMM_CONTROL_BIN_EDGES_US
        ],
        "burst_relative_histogram_mean_density_by_variant": density_by_variant,
        "workers": stats["flash_comm_burst_workers"],
    }


def _kernel_phase_bucket_rows(stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, row in stats["kernel_name_stats"].items():
        durations_ms = np.array(row["durations_us"], dtype=np.float64) / 1000.0
        count = int(row["count"])
        total_duration_s = float(row["duration_us"]) / 1e6
        phase_counts = row["heuristic_phase_counts"]
        kind = (
            row["kind_counts"].most_common(1)[0][0]
            if row["kind_counts"] else "unknown"
        )
        subcategory = (
            row["subcategory_counts"].most_common(1)[0][0]
            if row["subcategory_counts"] else "unknown"
        )

        p50_ms = _percentile(durations_ms, 50)
        p95_ms = _percentile(durations_ms, 95)
        max_ms = float(durations_ms.max()) if durations_ms.size else 0.0
        name_bucket, name_reason = _kernel_name_phase_rule(name)
        observed_bucket, observed_reason = _kernel_observed_phase_rule(
            name,
            count=count,
            p50_ms=p50_ms,
            p95_ms=p95_ms,
        )
        if name_bucket is not None:
            bucket = name_bucket
            reason = name_reason
            source = "name"
        elif observed_bucket is not None:
            bucket = observed_bucket
            reason = observed_reason
            source = "duration_shape"
        else:
            bucket = "unknown"
            reason = f"{name_reason}; {observed_reason}"
            source = "unknown"

        rows.append({
            "bucket": bucket,
            "source": source,
            "reason": reason,
            "kernel_name": name,
            "short_name": _short_kernel_name(name),
            "kind": kind,
            "subcategory": subcategory,
            "event_count": count,
            "duration_seconds": round(total_duration_s, 6),
            "p50_ms": round(float(p50_ms), 6),
            "p95_ms": round(float(p95_ms), 6),
            "max_ms": round(float(max_ms), 6),
            "heuristic_prefill_count": int(phase_counts.get("prefill", 0)),
            "heuristic_decode_count": int(phase_counts.get("decode", 0)),
            "heuristic_mixed_count": int(phase_counts.get("mixed", 0)),
            "heuristic_unknown_count": int(phase_counts.get("unknown", 0)),
        })
    return sorted(
        rows,
        key=lambda item: (
            KERNEL_PHASE_BUCKET_ORDER.index(item["bucket"]),
            -int(item["event_count"]),
            item["kernel_name"],
        ),
    )


def _kernel_phase_bucket_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    summary = {
        bucket: {
            "kernel_names": 0,
            "event_count": 0,
            "duration_seconds": 0.0,
        }
        for bucket in KERNEL_PHASE_BUCKET_ORDER
    }
    for row in rows:
        bucket = row["bucket"]
        summary[bucket]["kernel_names"] += 1
        summary[bucket]["event_count"] += int(row["event_count"])
        summary[bucket]["duration_seconds"] += float(row["duration_seconds"])
    for bucket in KERNEL_PHASE_BUCKET_ORDER:
        summary[bucket]["duration_seconds"] = round(
            summary[bucket]["duration_seconds"],
            6,
        )
    return summary


def _row_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if value in ("", None):
        return 0
    return int(value)


def _row_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    if value in ("", None):
        return 0.0
    return float(value)


def _manual_kernel_phase_rule(
    row: dict[str, Any],
) -> tuple[str, str, str]:
    """Best-effort phase label for kernel names.

    This is intentionally separate from the conservative bucket: it can use
    scheduler time-warp hints, but the output is marked heuristic.
    """
    conservative_bucket = str(row.get("bucket", "unknown"))
    if conservative_bucket in ("prefill", "decode"):
        return (
            conservative_bucket,
            f"conservative_{row.get('source', 'unknown')}",
            str(row.get("reason", "")),
        )

    name = str(row.get("kernel_name", ""))
    lower = name.lower()
    compact = lower.replace("_", "")
    subcategory = str(row.get("subcategory", "unknown"))

    explicit_prefill_patterns = (
        "prefill",
        "prompt",
        "context_attention",
        "contextattention",
        "context_fwd",
    )
    for pattern in explicit_prefill_patterns:
        if pattern in lower or pattern in compact:
            return "prefill", "manual_name", f"name pattern: {pattern}"

    explicit_decode_patterns = (
        "decode",
        "splitkv",
        "paged_attention",
        "pagedattention",
        "apply_penalty",
        "mask_top_p",
        "revert_output_bin_count",
        "multinomial",
        "argmax",
        "rejection",
    )
    for pattern in explicit_decode_patterns:
        if pattern in lower or pattern in compact:
            return "decode", "manual_name", f"name pattern: {pattern}"

    p50_ms = _row_float(row, "p50_ms")
    p95_ms = _row_float(row, "p95_ms")
    count = _row_int(row, "event_count")
    if _is_attention_like_kernel_name(name):
        if p50_ms >= 0.05 or p95_ms >= 0.20:
            return "prefill", "manual_duration_shape", (
                "attention-like kernel with long duration shape"
            )
        if count >= 100 and p95_ms <= 0.05:
            return "decode", "manual_duration_shape", (
                "attention-like kernel with short repeated duration shape"
            )

    sampling_decode_patterns = (
        "compare_scalar",
        "cunn_softmax",
        "softmax",
        "radix_sort",
        "sort_postprocess",
        "distribution_elementwise",
        "distributionnormal",
        "mask_top_p",
        "revert_output_bin_count",
    )
    if subcategory == "sampling_overhead" or any(
        pattern in lower or pattern in compact
        for pattern in sampling_decode_patterns
    ):
        return "decode", "manual_sampling", (
            "sampling/logit postprocess kernels occur during generation"
        )

    heuristic_prefill = _row_int(row, "heuristic_prefill_count")
    heuristic_decode = _row_int(row, "heuristic_decode_count")
    heuristic_mixed = _row_int(row, "heuristic_mixed_count")
    known_heuristic = heuristic_prefill + heuristic_decode + heuristic_mixed
    min_known = max(
        MANUAL_PHASE_HEURISTIC_MIN_EVENTS,
        int(count * 0.5),
    )
    if known_heuristic >= min_known:
        prefill_credit = heuristic_prefill + heuristic_mixed
        decode_credit = heuristic_decode + heuristic_mixed
        prefill_fraction = prefill_credit / known_heuristic
        decode_fraction = decode_credit / known_heuristic
        if (
            decode_credit > prefill_credit
            and decode_fraction >= MANUAL_PHASE_HEURISTIC_DOMINANCE
        ):
            return "decode", "scheduler_time_warp", (
                ">=90% of heuristic-mapped occurrences land in decode windows"
            )
        if (
            prefill_credit > decode_credit
            and prefill_fraction >= MANUAL_PHASE_HEURISTIC_DOMINANCE
        ):
            return "prefill", "scheduler_time_warp", (
                ">=90% of heuristic-mapped occurrences land in prefill windows"
            )

    return "unknown", "unknown", (
        "no explicit phase name, attention duration signal, sampling signal, "
        "or dominant scheduler-time heuristic"
    )


def _manual_kernel_phase_bucket_rows(
    conservative_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in conservative_rows:
        manual_bucket, manual_source, manual_reason = _manual_kernel_phase_rule(
            row
        )
        out = dict(row)
        out["manual_bucket"] = manual_bucket
        out["manual_source"] = manual_source
        out["manual_reason"] = manual_reason
        out["conservative_bucket"] = row.get("bucket", "unknown")
        out["conservative_source"] = row.get("source", "unknown")
        out["conservative_reason"] = row.get("reason", "")
        rows.append(out)
    return sorted(
        rows,
        key=lambda item: (
            KERNEL_PHASE_BUCKET_ORDER.index(item["manual_bucket"]),
            -_row_int(item, "event_count"),
            str(item.get("kernel_name", "")),
        ),
    )


def write_manual_kernel_phase_bucket_csv(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    if not rows:
        return
    fieldnames = [
        "manual_bucket",
        "manual_source",
        "manual_reason",
        "conservative_bucket",
        "conservative_source",
        "conservative_reason",
        "event_count",
        "duration_seconds",
        "p50_ms",
        "p95_ms",
        "max_ms",
        "heuristic_prefill_count",
        "heuristic_decode_count",
        "heuristic_mixed_count",
        "heuristic_unknown_count",
        "kind",
        "subcategory",
        "short_name",
        "kernel_name",
    ]
    with (out_dir / "manual_kernel_phase_buckets.csv").open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_kernel_phase_bucket_csv(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    if not rows:
        return
    fieldnames = [
        "bucket",
        "source",
        "reason",
        "event_count",
        "duration_seconds",
        "p50_ms",
        "p95_ms",
        "max_ms",
        "heuristic_prefill_count",
        "heuristic_decode_count",
        "heuristic_mixed_count",
        "heuristic_unknown_count",
        "kind",
        "subcategory",
        "short_name",
        "kernel_name",
    ]
    with (out_dir / "kernel_phase_buckets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(stats: dict[str, Any], out_dir: Path) -> None:
    kernel_bucket_rows = _kernel_phase_bucket_rows(stats)
    write_kernel_phase_bucket_csv(kernel_bucket_rows, out_dir)
    manual_kernel_bucket_rows = _manual_kernel_phase_bucket_rows(
        kernel_bucket_rows
    )
    write_manual_kernel_phase_bucket_csv(manual_kernel_bucket_rows, out_dir)
    payload: dict[str, Any] = {
        "label": stats["label"],
        "jsonl_files": stats["jsonl_files"],
        "nsys_rep_files": stats["nsys_rep_files"],
        "lines": stats["lines"],
        "decoded_rows": stats["decoded_rows"],
        "events": stats["events"],
        "skipped_rows": dict(stats["skipped_rows"]),
        "count_by_kind": dict(stats["count_by_kind"]),
        "duration_seconds_by_kind": _counter_json(
            stats["duration_us_by_kind"], scale=1e6
        ),
        "count_by_subcategory": dict(stats["count_by_subcategory"]),
        "duration_seconds_by_subcategory": _counter_json(
            stats["duration_us_by_subcategory"], scale=1e6
        ),
        "count_by_control_family": dict(stats["count_by_control_family"]),
        "duration_seconds_by_control_family": _counter_json(
            stats["duration_us_by_control_family"], scale=1e6
        ),
        "unclassified_count": int(sum(stats["unclassified"].values())),
        "top_unclassified": dict(stats["unclassified"].most_common(100)),
        "top_events_by_count": dict(stats["count_by_event"].most_common(50)),
        "top_events_by_duration_seconds": _counter_json(
            Counter(dict(stats["duration_us_by_event"].most_common(50))),
            scale=1e6,
        ),
        "note": (
            "Durations are summed CUPTI durations. Overlapping GPU streams and "
            "multiple workers are not de-overlapped."
        ),
        "event_phase_credit_note": (
            "Event prefill/decode attribution uses JSONL NVTX phase ranges only. "
            "Raw mixed events are credited to both prefill and decode in "
            "phase_count_by_kind and phase_duration_seconds_by_kind."
        ),
        "jsonl_phase_ranges": dict(stats["jsonl_phase_ranges"]),
        "jsonl_comm_phase_stats": dict(stats["jsonl_comm_phase_stats"]),
        "raw_event_phase_counts": dict(stats["raw_event_phase_counts"]),
        "phase_count_by_kind": {
            phase: dict(stats["phase_count_by_kind"][phase])
            for phase in EVENT_PHASE_ORDER
        },
        "phase_duration_seconds_by_kind": _nested_counter_json(
            stats["phase_duration_us_by_kind"],
            scale=1e6,
        ),
        "heuristic_event_phase_note": (
            "Heuristic phase attribution linearly warps each worker CUPTI trace "
            "time span onto the EngineCore scheduler elapsed timeline. This is "
            "a rough visualization only, not evidence of exact event phase."
        ),
        "heuristic_trace_windows": stats["heuristic_trace_windows"],
        "heuristic_raw_event_phase_counts": dict(
            stats["heuristic_raw_event_phase_counts"]
        ),
        "heuristic_phase_count_by_kind": {
            phase: dict(stats["heuristic_phase_count_by_kind"][phase])
            for phase in EVENT_PHASE_ORDER
        },
        "heuristic_phase_duration_seconds_by_kind": _nested_counter_json(
            stats["heuristic_phase_duration_us_by_kind"],
            scale=1e6,
        ),
        "control_comm_burst_cluster": _control_comm_burst_cluster_json(stats),
        "flash_kernel_summary": _flash_kernel_summary_json(stats),
        "kernel_phase_bucket_note": (
            "Conservative kernel-name inference. Strong name patterns are used "
            "first; attention-like kernels can be bucketed by duration shape. "
            "Generic GEMM/MoE/norm/elementwise kernels remain unknown unless "
            "their name carries a phase-specific signal."
        ),
        "kernel_phase_bucket_summary": _kernel_phase_bucket_summary(
            kernel_bucket_rows
        ),
        "top_kernel_phase_buckets": {
            bucket: [
                row for row in kernel_bucket_rows if row["bucket"] == bucket
            ][:50]
            for bucket in KERNEL_PHASE_BUCKET_ORDER
        },
        "manual_kernel_phase_bucket_note": (
            "Best-effort manual kernel phase inference. This uses explicit "
            "phase names, attention duration shape, sampling/logit "
            "postprocess patterns, then a labeled scheduler time-warp "
            "fallback. The scheduler fallback is approximate and should not "
            "be treated as ground-truth event attribution."
        ),
        "manual_kernel_phase_bucket_summary": _kernel_phase_bucket_summary(
            [
                {
                    **row,
                    "bucket": row["manual_bucket"],
                }
                for row in manual_kernel_bucket_rows
            ]
        ),
        "top_manual_kernel_phase_buckets": {
            bucket: [
                row for row in manual_kernel_bucket_rows
                if row["manual_bucket"] == bucket
            ][:50]
            for bucket in KERNEL_PHASE_BUCKET_ORDER
        },
        "scheduler_phase_source": (
            "EngineCore iteration logs. CUPTI events are not split by "
            "prefill/decode unless the JSONL export contains vLLM iteration or "
            "comm NVTX ranges."
        ),
        "iteration_logs": stats["iteration_logs"],
        "scheduler_iterations": stats["scheduler_iterations"],
        "scheduler_phase_counts": dict(stats["scheduler_phase_counts"]),
        "scheduler_phase_elapsed_seconds": _counter_json(
            stats["scheduler_phase_elapsed_ms"], scale=1000.0
        ),
        "scheduler_token_totals": dict(stats["scheduler_token_totals"]),
        "scheduler_context_tokens_by_phase": dict(
            stats["scheduler_context_tokens_by_phase"]
        ),
        "scheduler_generation_tokens_by_phase": dict(
            stats["scheduler_generation_tokens_by_phase"]
        ),
        "scheduler_token_summary": stats["scheduler_token_summary"],
        "workers": {},
    }
    for worker, worker_stats in sorted(stats["workers"].items()):
        payload["workers"][worker] = {
            "events": worker_stats["events"],
            "count_by_kind": dict(worker_stats["count_by_kind"]),
            "duration_seconds_by_kind": _counter_json(
                worker_stats["duration_us_by_kind"], scale=1e6
            ),
            "count_by_subcategory": dict(worker_stats["count_by_subcategory"]),
            "duration_seconds_by_subcategory": _counter_json(
                worker_stats["duration_us_by_subcategory"], scale=1e6
            ),
            "count_by_control_family": dict(
                worker_stats["count_by_control_family"]
            ),
            "duration_seconds_by_control_family": _counter_json(
                worker_stats["duration_us_by_control_family"], scale=1e6
            ),
            "unclassified_count": int(sum(worker_stats["unclassified"].values())),
            "top_unclassified": dict(worker_stats["unclassified"].most_common(50)),
        }

    (out_dir / "event_classification_summary.json").write_text(
        json.dumps(payload, indent=2)
    )
    lines = [
        f"# Event Classification Evaluation: {stats['label']}",
        "",
        f"- Events parsed: {stats['events']:,}",
        f"- Unclassified events: {sum(stats['unclassified'].values()):,}",
        f"- JSONL traces: {len(stats['jsonl_files'])}",
        "",
        "Durations are summed CUPTI durations and are not de-overlapped.",
        "",
    ]
    if stats["scheduler_iterations"]:
        token_summary = stats["scheduler_token_summary"]
        lines.extend([
            f"- Scheduler iterations: {stats['scheduler_iterations']:,}",
            (
                "- Scheduler token split: "
                f"{token_summary.get('prefill_tokens_total', 0):,} prefill / "
                f"{token_summary.get('decode_tokens_total', 0):,} decode"
            ),
            "",
        ])
    if stats["events"]:
        raw_phases = stats["raw_event_phase_counts"]
        phase_known = sum(raw_phases.get(p, 0) for p in ("prefill", "decode", "mixed"))
        lines.extend([
            (
                "- Event phase ranges: "
                f"{sum(stats['jsonl_phase_ranges'].values()):,} NVTX ranges"
            ),
            (
                "- Event phase attribution: "
                f"{phase_known:,} known / {raw_phases.get('unknown', 0):,} unknown"
            ),
            "",
        ])
        if stats["heuristic_raw_event_phase_counts"]:
            h_counts = stats["heuristic_raw_event_phase_counts"]
            h_known = sum(
                h_counts.get(p, 0) for p in ("prefill", "decode", "mixed")
            )
            lines.extend([
                (
                    "- Heuristic event phase attribution: "
                    f"{h_known:,} known / {h_counts.get('unknown', 0):,} unknown"
                ),
                (
                    "  Heuristic attribution linearly warps CUPTI trace time "
                    "onto scheduler elapsed time; use only as rough visualization."
                ),
                "",
            ])
    if stats["events"]:
        cluster = stats["control_comm_burst_cluster"]
        control_events = int(cluster.get("control_events", 0))
        if control_events > 0:
            clustered_controls = int(cluster.get("clustered_control_events", 0))
            fraction = clustered_controls / control_events
            lines.extend([
                (
                    "- Control/comm boundary clustering (Fig. 4 style, all phases): "
                    f"{clustered_controls:,} / {control_events:,} control events "
                    f"within +/-5 ms of a comm burst start ({fraction:.1%}); "
                    f"{int(cluster.get('comm_bursts', 0)):,} merged comm bursts"
                ),
                (
                    "  Phase-agnostic: all classified control/comm events. "
                    "X-axis is time relative to comm burst start (0=burst)."
                ),
                "",
            ])
        flash_events = stats["flash_kernel_events"]
        if flash_events:
            flash_summary = _flash_kernel_summary_json(stats)
            main = flash_summary["duration_ms_by_variant"]["run_flash_fwd"]
            lines.extend([
                (
                    "- Flash forward kernels: "
                    f"{flash_summary['flash_events']:,} matched events; "
                    f"run_flash_fwd p50/p95/max = "
                    f"{main['p50_ms']:.3f}/{main['p95_ms']:.3f}/"
                    f"{main['max_ms']:.3f} ms"
                ),
                (
                    "  Matched by event name; plots use time relative to comm "
                    "burst start where applicable."
                ),
                "",
            ])
        if kernel_bucket_rows:
            bucket_summary = _kernel_phase_bucket_summary(kernel_bucket_rows)
            parts = []
            for bucket in KERNEL_PHASE_BUCKET_ORDER:
                item = bucket_summary[bucket]
                parts.append(
                    f"{bucket}: {int(item['event_count']):,} events / "
                    f"{int(item['kernel_names']):,} names"
                )
            lines.extend([
                "- Kernel-name phase buckets: " + "; ".join(parts),
                (
                    "  Conservative inference; see kernel_phase_buckets.csv "
                    "for every kernel name and rule reason."
                ),
                "",
            ])
    if not stats["jsonl_files"]:
        lines.append("No JSONL traces were available for this experiment.")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def write_scheduler_summary(
    stats: dict[str, Any],
    iterations: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    if not iterations:
        return
    payload = {
        "source": "EngineCore iteration logs",
        "phase_note": (
            "prefill_tokens are context_tokens; decode_tokens are "
            "generation_tokens. mixed iterations contain both."
        ),
        "iteration_logs": stats["iteration_logs"],
        "summary": stats["scheduler_token_summary"],
        "phase_counts": dict(stats["scheduler_phase_counts"]),
        "phase_elapsed_seconds": _counter_json(
            stats["scheduler_phase_elapsed_ms"], scale=1000.0
        ),
        "token_totals": dict(stats["scheduler_token_totals"]),
        "context_tokens_by_phase": dict(
            stats["scheduler_context_tokens_by_phase"]
        ),
        "generation_tokens_by_phase": dict(
            stats["scheduler_generation_tokens_by_phase"]
        ),
        "iterations": iterations,
    }
    (out_dir / "scheduler_prefill_decode_summary.json").write_text(
        json.dumps(payload, indent=2)
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_simple_bars(
    values: Counter,
    out_dir: Path,
    stem: str,
    title: str,
    ylabel: str,
    *,
    order: tuple[str, ...] | None = None,
    scale: float = 1.0,
) -> None:
    if not values:
        return
    keys = list(order or values.keys())
    keys.extend(k for k in values if k not in keys)
    keys = [k for k in keys if values.get(k, 0)]
    vals = [values[k] / scale for k in keys]
    colors = [KIND_COLORS.get(k, "#777777") for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.2), 5))
    ax.bar(keys, vals, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:,.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def plot_horizontal_top(
    values: Counter,
    out_dir: Path,
    stem: str,
    title: str,
    xlabel: str,
    *,
    scale: float = 1.0,
    top_n: int = 25,
) -> None:
    top = [(k, v / scale) for k, v in values.most_common(top_n) if v]
    if not top:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No entries", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        ax.set_title(title)
        _save(fig, out_dir, stem)
        return
    labels = [k for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]

    fig, ax = plt.subplots(figsize=(13, max(5, 0.4 * len(labels) + 1.5)))
    ax.barh(range(len(labels)), vals, color="#4e79a7")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    for i, val in enumerate(vals):
        ax.text(val, i, f" {val:,.2f}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def plot_worker_kind_stack(
    stats: dict[str, Any],
    out_dir: Path,
    *,
    duration: bool,
) -> None:
    workers = sorted(stats["workers"])
    if not workers:
        return
    key = "duration_us_by_kind" if duration else "count_by_kind"
    scale = 1e6 if duration else 1.0
    ylabel = "Summed duration (s)" if duration else "Event count"
    stem = "worker_kind_duration_stacked" if duration else "worker_kind_count_stacked"
    title = "Control/Comm/Compute Duration by Worker" if duration else (
        "Control/Comm/Compute Count by Worker"
    )

    x = np.arange(len(workers))
    bottoms = np.zeros(len(workers))
    fig, ax = plt.subplots(figsize=(max(8, len(workers) * 1.2), 5))
    for kind in KIND_ORDER:
        vals = np.array([
            stats["workers"][worker][key].get(kind, 0) / scale
            for worker in workers
        ])
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=kind,
            color=KIND_COLORS[kind],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(workers, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def plot_duration_hist(stats: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = np.sqrt(HIST_BINS_US[:-1] * HIST_BINS_US[1:])
    for kind in KIND_ORDER:
        counts = stats["hist_by_kind"][kind]
        if counts.sum() == 0:
            continue
        ax.plot(
            centers,
            counts,
            drawstyle="steps-mid",
            label=kind,
            color=KIND_COLORS[kind],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Event duration (us)")
    ax.set_ylabel("Event count")
    ax.set_title("Duration Histogram by Kind")
    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "duration_histogram_by_kind")


def plot_raw_event_phase_counts(stats: dict[str, Any], out_dir: Path) -> None:
    values = stats["raw_event_phase_counts"]
    keys = [phase for phase in RAW_EVENT_PHASE_ORDER if values.get(phase, 0)]
    if not keys:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.3), 4.5))
    vals = [float(values[phase]) for phase in keys]
    ax.bar(
        [phase for phase in keys],
        vals,
        color=[EVENT_PHASE_COLORS[phase] for phase in keys],
    )
    ax.set_title("Raw Event Phase Attribution")
    ax.set_ylabel("Event count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "raw_event_phase_counts")


def plot_heuristic_raw_event_phase_counts(
    stats: dict[str, Any],
    out_dir: Path,
) -> None:
    values = stats["heuristic_raw_event_phase_counts"]
    keys = [phase for phase in RAW_EVENT_PHASE_ORDER if values.get(phase, 0)]
    if not keys:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.3), 4.5))
    vals = [float(values[phase]) for phase in keys]
    ax.bar(
        [phase for phase in keys],
        vals,
        color=[EVENT_PHASE_COLORS[phase] for phase in keys],
    )
    ax.set_title("Heuristic Event Phase Attribution")
    ax.set_ylabel("Event count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.5,
        -0.22,
        "HEURISTIC: CUPTI trace time linearly warped onto scheduler elapsed time",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, out_dir, "heuristic_raw_event_phase_counts")


def plot_phase_kind_stack(
    stats: dict[str, Any],
    out_dir: Path,
    *,
    duration: bool,
) -> None:
    key = "phase_duration_us_by_kind" if duration else "phase_count_by_kind"
    scale = 1e6 if duration else 1.0
    ylabel = "Summed duration (s)" if duration else "Event count"
    stem = "phase_kind_duration_stacked" if duration else (
        "phase_kind_count_stacked"
    )
    title = "Control/Comm/Compute by Event Phase"
    if duration:
        title = "Control/Comm/Compute Duration by Event Phase"

    phase_counters = stats[key]
    if not any(phase_counters[phase] for phase in EVENT_PHASE_ORDER):
        return

    x = np.arange(len(EVENT_PHASE_ORDER))
    bottoms = np.zeros(len(EVENT_PHASE_ORDER))
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind in KIND_ORDER:
        vals = np.array([
            phase_counters[phase].get(kind, 0) / scale
            for phase in EVENT_PHASE_ORDER
        ])
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=kind,
            color=KIND_COLORS[kind],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(EVENT_PHASE_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.5,
        -0.20,
        "mixed events are credited to both prefill and decode",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def plot_heuristic_phase_kind_stack(
    stats: dict[str, Any],
    out_dir: Path,
    *,
    duration: bool,
) -> None:
    key = (
        "heuristic_phase_duration_us_by_kind"
        if duration else "heuristic_phase_count_by_kind"
    )
    scale = 1e6 if duration else 1.0
    ylabel = "Summed duration (s)" if duration else "Event count"
    stem = "heuristic_phase_kind_duration_stacked" if duration else (
        "heuristic_phase_kind_count_stacked"
    )
    title = "Heuristic Control/Comm/Compute by Event Phase"
    if duration:
        title = "Heuristic Control/Comm/Compute Duration by Event Phase"

    phase_counters = stats[key]
    if not any(phase_counters[phase] for phase in EVENT_PHASE_ORDER):
        return

    x = np.arange(len(EVENT_PHASE_ORDER))
    bottoms = np.zeros(len(EVENT_PHASE_ORDER))
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind in KIND_ORDER:
        vals = np.array([
            phase_counters[phase].get(kind, 0) / scale
            for phase in EVENT_PHASE_ORDER
        ])
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=kind,
            color=KIND_COLORS[kind],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(EVENT_PHASE_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.5,
        -0.24,
        "HEURISTIC: linear time-warp; mixed events credited to both phases",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def _cluster_json_from_summary(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("control_comm_burst_cluster") or data.get(
        "heuristic_prefill_control_comm_cluster",
        {},
    )


def plot_control_clusters_at_boundaries(
    stats: dict[str, Any],
    out_dir: Path,
) -> None:
    cluster = stats["control_comm_burst_cluster"]
    comm_bursts = int(cluster.get("comm_bursts", 0))
    if comm_bursts <= 0:
        return

    hist_sum = stats["control_comm_burst_density_hist"]
    hist = hist_sum / comm_bursts
    edges_ms = PREFILL_COMM_CONTROL_BIN_EDGES_US / 1000.0
    centers_ms = (edges_ms[:-1] + edges_ms[1:]) / 2.0
    widths_ms = np.diff(edges_ms)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(
        centers_ms,
        hist,
        width=widths_ms * 0.95,
        color="#4e79a7",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        0,
        color="#d62728",
        linewidth=1.2,
        linestyle="--",
        label="burst start",
    )
    ax.set_title("Control clusters at boundaries")
    ax.set_xlabel("time relative to communication burst (ms)")
    ax.set_ylabel("mean control events / bin")
    ax.set_xlim(edges_ms[0], edges_ms[-1])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.5,
        -0.18,
        (
            "Comm events merged within "
            f"{PREFILL_COMM_BURST_MERGE_GAP_US / 1000:.1f} ms; "
            "density averaged over bursts"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, out_dir, "control_clusters_at_boundaries")


def plot_control_comm_burst_density(
    stats: dict[str, Any],
    out_dir: Path,
) -> None:
    cluster = stats["control_comm_burst_cluster"]
    control_events = int(cluster.get("control_events", 0))
    comm_bursts = int(cluster.get("comm_bursts", 0))
    if control_events <= 0 or comm_bursts <= 0:
        return

    hist_sum = stats["control_comm_burst_density_hist"]
    hist = hist_sum / comm_bursts
    edges_ms = PREFILL_COMM_CONTROL_BIN_EDGES_US / 1000.0
    centers_ms = (edges_ms[:-1] + edges_ms[1:]) / 2.0
    widths_ms = np.diff(edges_ms)
    mean_burst_duration_ms = (
        float(cluster.get("burst_duration_us_sum", 0)) / comm_bursts / 1000.0
    )
    mean_per_burst = _control_comm_burst_cluster_json(stats)[
        "mean_control_per_burst"
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )
    ax = axes[0]
    ax.bar(
        centers_ms,
        hist,
        width=widths_ms * 0.95,
        color="#59a14f",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--", label="burst start")
    if mean_burst_duration_ms > 0:
        ax.axvspan(
            0,
            mean_burst_duration_ms,
            color="#e15759",
            alpha=0.15,
            label="mean comm burst",
        )
    ax.set_title("Control Density vs Comm Burst (Fig. 4)")
    ax.set_xlabel("Time relative to comm burst start (ms)")
    ax.set_ylabel("Mean control events per burst")
    ax.set_xlim(edges_ms[0], edges_ms[-1])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["pre-burst", "during", "post-burst"]
    vals = [
        float(mean_per_burst["pre_burst_window"]),
        float(mean_per_burst["during_burst"]),
        float(mean_per_burst["post_burst_window"]),
    ]
    colors = ["#4e79a7", "#e15759", "#59a14f"]
    ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_title("Mean Control per Burst by Window")
    ax.set_ylabel("Control events / burst")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    fig.text(
        0.5,
        0.01,
        (
            "All phases; 0=comm burst start; "
            "density averaged over bursts in +/-5 ms window"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out_dir, "control_comm_burst_density")


def plot_flash_kernel_timeline(stats: dict[str, Any], out_dir: Path) -> None:
    events = stats["flash_kernel_events"]
    if not events:
        return

    start0 = min(int(event["start_us"]) for event in events)
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant in FLASH_KERNEL_VARIANT_ORDER:
        xs = [
            (int(event["start_us"]) - start0) / 1e6
            for event in events
            if event["variant"] == variant
        ]
        ys = [
            max(float(event["dur_us"]) / 1000.0, 1e-4)
            for event in events
            if event["variant"] == variant
        ]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=8,
            alpha=0.45,
            label=FLASH_KERNEL_LABELS[variant],
            color=FLASH_KERNEL_COLORS[variant],
            linewidths=0,
        )
    ax.set_yscale("log")
    ax.set_title("Flash Forward Kernel Durations over Trace Time")
    ax.set_xlabel("Trace time from first matched event (s)")
    ax.set_ylabel("Kernel duration (ms, log)")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(
        0.5,
        -0.18,
        "Matched by event name: run_flash_fwd / flash_fwd_kernel equivalents",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, out_dir, "flash_fwd_kernel_timeline")


def plot_flash_kernel_duration_hist(stats: dict[str, Any], out_dir: Path) -> None:
    events = stats["flash_kernel_events"]
    if not events:
        return

    durations_ms = np.array(
        [max(float(event["dur_us"]) / 1000.0, 1e-4) for event in events],
        dtype=np.float64,
    )
    bins = np.logspace(
        np.log10(max(float(durations_ms.min()), 1e-4)),
        np.log10(max(float(durations_ms.max()), 1e-3)),
        50,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in FLASH_KERNEL_VARIANT_ORDER:
        vals = np.array(
            [
                max(float(event["dur_us"]) / 1000.0, 1e-4)
                for event in events
                if event["variant"] == variant
            ],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            label=FLASH_KERNEL_LABELS[variant],
            color=FLASH_KERNEL_COLORS[variant],
        )
    ax.set_xscale("log")
    ax.set_title("Flash Forward Kernel Duration Distribution")
    ax.set_xlabel("Kernel duration (ms, log)")
    ax.set_ylabel("Event count")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "flash_fwd_kernel_duration_hist")


def plot_flash_kernel_comm_burst_density(
    stats: dict[str, Any],
    out_dir: Path,
) -> None:
    cluster = stats["flash_comm_burst_cluster"]
    comm_bursts = int(cluster.get("comm_bursts", 0))
    if comm_bursts <= 0 or not stats["flash_kernel_events"]:
        return

    edges_ms = PREFILL_COMM_CONTROL_BIN_EDGES_US / 1000.0
    centers_ms = (edges_ms[:-1] + edges_ms[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in FLASH_KERNEL_VARIANT_ORDER:
        hist = stats["flash_comm_burst_density_hist_by_variant"][variant]
        if hist.sum() <= 0:
            continue
        ax.plot(
            centers_ms,
            hist / comm_bursts,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=FLASH_KERNEL_LABELS[variant],
            color=FLASH_KERNEL_COLORS[variant],
        )
    ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_title("Flash Forward Kernel Density vs Comm Burst")
    ax.set_xlabel("Time relative to comm burst start (ms)")
    ax.set_ylabel("Mean flash kernel events per burst")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(
        0.5,
        -0.18,
        "All phases; 0=comm burst start; density averaged over bursts",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, out_dir, "flash_fwd_kernel_comm_burst_density")


def plot_kernel_phase_bucket_summary(
    stats: dict[str, Any],
    out_dir: Path,
) -> None:
    rows = _kernel_phase_bucket_rows(stats)
    if not rows:
        return
    summary = _kernel_phase_bucket_summary(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, ylabel, title in (
        (
            axes[0],
            "event_count",
            "Kernel events",
            "Kernel Events by Inferred Phase Bucket",
        ),
        (
            axes[1],
            "duration_seconds",
            "Summed duration (s)",
            "Kernel Duration by Inferred Phase Bucket",
        ),
    ):
        vals = [float(summary[bucket][key]) for bucket in KERNEL_PHASE_BUCKET_ORDER]
        colors = [
            KERNEL_PHASE_BUCKET_COLORS[bucket]
            for bucket in KERNEL_PHASE_BUCKET_ORDER
        ]
        ax.bar(KERNEL_PHASE_BUCKET_ORDER, vals, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        for idx, value in enumerate(vals):
            label = f"{value:,.2f}" if key == "duration_seconds" else f"{value:,.0f}"
            ax.text(idx, value, label, ha="center", va="bottom", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Conservative name/duration inference; generic kernels stay unknown",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out_dir, "kernel_phase_bucket_summary")


def plot_top_kernel_phase_buckets(
    stats: dict[str, Any],
    out_dir: Path,
    *,
    top_n: int = 20,
) -> None:
    rows = _kernel_phase_bucket_rows(stats)
    if not rows:
        return
    for bucket in KERNEL_PHASE_BUCKET_ORDER:
        bucket_rows = [
            row for row in rows
            if row["bucket"] == bucket and int(row["event_count"]) > 0
        ][:top_n]
        values = Counter({
            (
                f"{row['short_name']} p50={row['p50_ms']:.3g}ms "
                f"({row['source']})"
            ): int(row["event_count"])
            for row in bucket_rows
        })
        plot_horizontal_top(
            values,
            out_dir,
            f"top_{bucket}_kernel_names",
            f"Top {bucket.title()} Kernel Names",
            "Kernel event count",
            top_n=top_n,
        )


def plot_manual_kernel_phase_bucket_summary(
    stats: dict[str, Any],
    out_dir: Path,
) -> None:
    rows = _manual_kernel_phase_bucket_rows(_kernel_phase_bucket_rows(stats))
    if not rows:
        return
    summary = _kernel_phase_bucket_summary([
        {**row, "bucket": row["manual_bucket"]}
        for row in rows
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, key, ylabel, title in (
        (
            axes[0],
            "event_count",
            "Kernel events",
            "Kernel Events by Manual Phase Bucket",
        ),
        (
            axes[1],
            "duration_seconds",
            "Summed duration (s)",
            "Kernel Duration by Manual Phase Bucket",
        ),
    ):
        vals = [float(summary[bucket][key]) for bucket in KERNEL_PHASE_BUCKET_ORDER]
        colors = [
            KERNEL_PHASE_BUCKET_COLORS[bucket]
            for bucket in KERNEL_PHASE_BUCKET_ORDER
        ]
        ax.bar(KERNEL_PHASE_BUCKET_ORDER, vals, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        for idx, value in enumerate(vals):
            label = f"{value:,.2f}" if key == "duration_seconds" else f"{value:,.0f}"
            ax.text(idx, value, label, ha="center", va="bottom", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Best-effort rules; scheduler time-warp fallback is approximate",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out_dir, "manual_kernel_phase_bucket_summary")


def plot_top_manual_kernel_phase_buckets(
    stats: dict[str, Any],
    out_dir: Path,
    *,
    top_n: int = 20,
) -> None:
    rows = _manual_kernel_phase_bucket_rows(_kernel_phase_bucket_rows(stats))
    if not rows:
        return
    for bucket in KERNEL_PHASE_BUCKET_ORDER:
        bucket_rows = [
            row for row in rows
            if row["manual_bucket"] == bucket and int(row["event_count"]) > 0
        ][:top_n]
        values = Counter({
            (
                f"{row['short_name']} p50={row['p50_ms']:.3g}ms "
                f"({row['manual_source']})"
            ): int(row["event_count"])
            for row in bucket_rows
        })
        plot_horizontal_top(
            values,
            out_dir,
            f"top_manual_{bucket}_kernel_names",
            f"Top Manual {bucket.title()} Kernel Names",
            "Kernel event count",
            top_n=top_n,
        )


def plot_scheduler_phase_bar(
    values: Counter,
    out_dir: Path,
    stem: str,
    title: str,
    ylabel: str,
    *,
    scale: float = 1.0,
) -> None:
    keys = [
        phase
        for phase in SCHEDULER_PHASE_ORDER
        if float(values.get(phase, 0.0)) > 0.0
    ]
    if not keys:
        return
    labels = [SCHEDULER_PHASE_LABELS[k] for k in keys]
    vals = [float(values[k]) / scale for k in keys]
    colors = [SCHEDULER_PHASE_COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.4), 4.5))
    ax.bar(labels, vals, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    for i, val in enumerate(vals):
        label = f"{val:,.2f}" if scale != 1.0 else f"{val:,.0f}"
        ax.text(i, val, label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def plot_scheduler_token_totals(stats: dict[str, Any], out_dir: Path) -> None:
    values = stats["scheduler_token_totals"]
    keys = [key for key in TOKEN_SPLIT_ORDER if values.get(key, 0)]
    if not keys:
        return
    labels = [TOKEN_SPLIT_LABELS[key] for key in keys]
    vals = [float(values[key]) for key in keys]
    colors = [TOKEN_SPLIT_COLORS[key] for key in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, vals, color=colors)
    ax.set_title("Scheduler Prefill vs Decode Token Totals")
    ax.set_ylabel("Tokens")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=10)
    for i, val in enumerate(vals):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "scheduler_prefill_decode_token_totals")


def plot_scheduler_tokens_by_phase(stats: dict[str, Any], out_dir: Path) -> None:
    context = stats["scheduler_context_tokens_by_phase"]
    generation = stats["scheduler_generation_tokens_by_phase"]
    keys = [
        phase
        for phase in SCHEDULER_PHASE_ORDER
        if context.get(phase, 0) or generation.get(phase, 0)
    ]
    if not keys:
        return
    x = np.arange(len(keys))
    ctx = np.array([float(context.get(key, 0)) for key in keys])
    gen = np.array([float(generation.get(key, 0)) for key in keys])

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.4), 4.8))
    ax.bar(
        x,
        gen,
        label="decode/generation tokens",
        color=TOKEN_SPLIT_COLORS["decode_tokens"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.bar(
        x,
        ctx,
        bottom=gen,
        label="prefill/context tokens",
        color=TOKEN_SPLIT_COLORS["prefill_tokens"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEDULER_PHASE_LABELS[key] for key in keys], rotation=15)
    ax.set_title("Scheduler Tokens by Iteration Phase")
    ax.set_ylabel("Tokens")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "scheduler_tokens_by_phase")


def plot_scheduler_token_timeline(
    iterations: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_num_batched_tokens: int | None,
) -> None:
    if not iterations:
        return

    x = np.array([int(it["iteration"]) for it in iterations], dtype=np.int64)
    ctx = np.array(
        [int(it.get("context_tokens", 0)) for it in iterations],
        dtype=np.float64,
    )
    gen = np.array(
        [int(it.get("generation_tokens", 0)) for it in iterations],
        dtype=np.float64,
    )
    total = ctx + gen

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.fill_between(
        x,
        0,
        gen,
        step="mid",
        alpha=0.70,
        color=TOKEN_SPLIT_COLORS["decode_tokens"],
        label="decode/generation tokens",
    )
    ax.fill_between(
        x,
        gen,
        total,
        step="mid",
        alpha=0.75,
        color=TOKEN_SPLIT_COLORS["prefill_tokens"],
        label="prefill/context tokens",
    )
    ax.plot(x, total, color="#2f4858", linewidth=0.9, label="total tokens")
    if max_num_batched_tokens:
        ax.axhline(
            max_num_batched_tokens,
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label=f"max batched tokens={max_num_batched_tokens}",
        )
    ax.set_title("Scheduler Prefill/Decode Tokens per Iteration")
    ax.set_xlabel("Scheduler iteration")
    ax.set_ylabel("Tokens")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "scheduler_tokens_per_iteration")


def plot_scheduler_prefill_zoom(
    iterations: list[dict[str, Any]],
    out_dir: Path,
    *,
    context_iters: int = 30,
) -> None:
    if not iterations:
        return

    x = np.array([int(it["iteration"]) for it in iterations], dtype=np.int64)
    ctx = np.array(
        [int(it.get("context_tokens", 0)) for it in iterations],
        dtype=np.float64,
    )
    gen = np.array(
        [int(it.get("generation_tokens", 0)) for it in iterations],
        dtype=np.float64,
    )
    prefill_idx = np.where(ctx > 0)[0]
    if prefill_idx.size == 0:
        return
    lo = max(0, int(prefill_idx.min()) - context_iters)
    hi = min(len(x), int(prefill_idx.max()) + context_iters + 1)
    xz = x[lo:hi]
    ctxz = ctx[lo:hi]
    genz = gen[lo:hi]
    total = ctxz + genz

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.fill_between(
        xz,
        0,
        genz,
        step="mid",
        alpha=0.65,
        color=TOKEN_SPLIT_COLORS["decode_tokens"],
        label="decode/generation tokens",
    )
    ax.fill_between(
        xz,
        genz,
        total,
        step="mid",
        alpha=0.80,
        color=TOKEN_SPLIT_COLORS["prefill_tokens"],
        label="prefill/context tokens",
    )
    active = np.where(ctxz > 0)[0]
    ax.scatter(
        xz[active],
        total[active],
        s=12,
        color="#2f4858",
        zorder=5,
        label="prefill-active iterations",
    )
    ax.set_title("Scheduler Prefill-Active Window")
    ax.set_xlabel("Scheduler iteration")
    ax.set_ylabel("Tokens")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "scheduler_prefill_zoom")


def plot_scheduler_all(
    stats: dict[str, Any],
    iterations: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_num_batched_tokens: int | None,
) -> None:
    if not iterations:
        return
    plot_scheduler_phase_bar(
        stats["scheduler_phase_counts"],
        out_dir,
        "scheduler_iteration_phase_counts",
        "Scheduler Iteration Counts by Phase",
        "Iterations",
    )
    plot_scheduler_phase_bar(
        stats["scheduler_phase_elapsed_ms"],
        out_dir,
        "scheduler_iteration_phase_elapsed",
        "Scheduler Elapsed Time by Phase",
        "Elapsed time (s)",
        scale=1000.0,
    )
    plot_scheduler_token_totals(stats, out_dir)
    plot_scheduler_tokens_by_phase(stats, out_dir)
    plot_scheduler_token_timeline(
        iterations,
        out_dir,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    plot_scheduler_prefill_zoom(iterations, out_dir)


def plot_all(stats: dict[str, Any], out_dir: Path) -> None:
    plot_simple_bars(
        stats["count_by_kind"],
        out_dir,
        "kind_event_counts",
        "Control/Comm/Compute Event Counts",
        "Event count",
        order=KIND_ORDER,
    )
    plot_simple_bars(
        stats["duration_us_by_kind"],
        out_dir,
        "kind_event_duration",
        "Control/Comm/Compute Summed Durations",
        "Summed duration (s)",
        order=KIND_ORDER,
        scale=1e6,
    )
    plot_horizontal_top(
        stats["duration_us_by_subcategory"],
        out_dir,
        "subcategory_duration",
        "Duration by Event Subcategory",
        "Summed duration (s)",
        scale=1e6,
        top_n=len(SUB_ORDER),
    )
    plot_horizontal_top(
        stats["duration_us_by_control_family"],
        out_dir,
        "control_family_duration",
        "Control Duration by Family",
        "Summed duration (s)",
        scale=1e6,
        top_n=len(CONTROL_FAMILY_ORDER),
    )
    plot_horizontal_top(
        stats["count_by_control_family"],
        out_dir,
        "control_family_counts",
        "Control Event Counts by Family",
        "Event count",
        top_n=len(CONTROL_FAMILY_ORDER),
    )
    plot_horizontal_top(
        stats["duration_us_by_event"],
        out_dir,
        "top_events_by_duration",
        "Top Events by Summed Duration",
        "Summed duration (s)",
        scale=1e6,
    )
    plot_horizontal_top(
        stats["count_by_event"],
        out_dir,
        "top_events_by_count",
        "Top Events by Count",
        "Event count",
    )
    plot_horizontal_top(
        stats["unclassified"],
        out_dir,
        "top_unclassified",
        "Top Unclassified Events",
        "Event count",
    )
    plot_worker_kind_stack(stats, out_dir, duration=True)
    plot_worker_kind_stack(stats, out_dir, duration=False)
    plot_duration_hist(stats, out_dir)
    plot_raw_event_phase_counts(stats, out_dir)
    plot_phase_kind_stack(stats, out_dir, duration=False)
    plot_phase_kind_stack(stats, out_dir, duration=True)
    plot_heuristic_raw_event_phase_counts(stats, out_dir)
    plot_heuristic_phase_kind_stack(stats, out_dir, duration=False)
    plot_heuristic_phase_kind_stack(stats, out_dir, duration=True)
    plot_control_clusters_at_boundaries(stats, out_dir)
    plot_control_comm_burst_density(stats, out_dir)
    plot_flash_kernel_timeline(stats, out_dir)
    plot_flash_kernel_duration_hist(stats, out_dir)
    plot_flash_kernel_comm_burst_density(stats, out_dir)
    plot_kernel_phase_bucket_summary(stats, out_dir)
    plot_top_kernel_phase_buckets(stats, out_dir)
    plot_manual_kernel_phase_bucket_summary(stats, out_dir)
    plot_top_manual_kernel_phase_buckets(stats, out_dir)


def plot_cross_experiment_summary(
    index: list[dict[str, Any]],
    summary_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for item in index:
        summary_path = Path(item["out_dir"]) / "event_classification_summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text())
        if not data.get("events"):
            continue
        rows.append({
            "label": item["label"],
            "count_by_kind": data.get("count_by_kind", {}),
            "duration_seconds_by_kind": data.get("duration_seconds_by_kind", {}),
            "phase_count_by_kind": data.get("phase_count_by_kind", {}),
            "phase_duration_seconds_by_kind": data.get(
                "phase_duration_seconds_by_kind", {}
            ),
            "heuristic_phase_count_by_kind": data.get(
                "heuristic_phase_count_by_kind", {}
            ),
            "heuristic_phase_duration_seconds_by_kind": data.get(
                "heuristic_phase_duration_seconds_by_kind", {}
            ),
            "control_comm_burst_cluster": _cluster_json_from_summary(data),
            "flash_kernel_summary": data.get("flash_kernel_summary", {}),
            "kernel_phase_bucket_summary": data.get(
                "kernel_phase_bucket_summary", {}
            ),
            "manual_kernel_phase_bucket_summary": data.get(
                "manual_kernel_phase_bucket_summary", {}
            ),
            "unclassified_count": data.get("unclassified_count", 0),
        })
    if not rows:
        return

    for key, ylabel, stem, title in (
        (
            "count_by_kind",
            "Event count",
            "kind_count_by_experiment",
            "Control/Comm/Compute Counts by Experiment",
        ),
        (
            "duration_seconds_by_kind",
            "Summed duration (s)",
            "kind_duration_by_experiment",
            "Control/Comm/Compute Durations by Experiment",
        ),
    ):
        x = np.arange(len(rows))
        bottoms = np.zeros(len(rows))
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
        for kind in KIND_ORDER:
            vals = np.array([
                float(row[key].get(kind, 0.0))
                for row in rows
            ])
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=kind,
                color=KIND_COLORS[kind],
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in rows], rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        _save(fig, summary_dir, stem)

    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.4), 4))
    vals = [row["unclassified_count"] for row in rows]
    ax.bar([row["label"] for row in rows], vals, color="#4e79a7")
    ax.set_title("Unclassified Events by Experiment")
    ax.set_ylabel("Event count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, summary_dir, "unclassified_by_experiment")

    for metric, ylabel, stem, title in (
        (
            "event_count",
            "Kernel events",
            "kernel_phase_bucket_events_by_experiment",
            "Kernel Events by Inferred Phase Bucket",
        ),
        (
            "duration_seconds",
            "Summed duration (s)",
            "kernel_phase_bucket_duration_by_experiment",
            "Kernel Duration by Inferred Phase Bucket",
        ),
    ):
        if not any(row["kernel_phase_bucket_summary"] for row in rows):
            continue
        x = np.arange(len(rows))
        bottoms = np.zeros(len(rows))
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
        for bucket in KERNEL_PHASE_BUCKET_ORDER:
            vals = np.array([
                float(
                    row["kernel_phase_bucket_summary"]
                    .get(bucket, {})
                    .get(metric, 0.0)
                )
                for row in rows
            ])
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=bucket,
                color=KERNEL_PHASE_BUCKET_COLORS[bucket],
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in rows], rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.text(
            0.5,
            -0.24,
            "Conservative name/duration inference; generic kernels stay unknown",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        _save(fig, summary_dir, stem)

    for metric, ylabel, stem, title in (
        (
            "event_count",
            "Kernel events",
            "manual_kernel_phase_bucket_events_by_experiment",
            "Kernel Events by Manual Phase Bucket",
        ),
        (
            "duration_seconds",
            "Summed duration (s)",
            "manual_kernel_phase_bucket_duration_by_experiment",
            "Kernel Duration by Manual Phase Bucket",
        ),
    ):
        if not any(row["manual_kernel_phase_bucket_summary"] for row in rows):
            continue
        x = np.arange(len(rows))
        bottoms = np.zeros(len(rows))
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
        for bucket in KERNEL_PHASE_BUCKET_ORDER:
            vals = np.array([
                float(
                    row["manual_kernel_phase_bucket_summary"]
                    .get(bucket, {})
                    .get(metric, 0.0)
                )
                for row in rows
            ])
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=bucket,
                color=KERNEL_PHASE_BUCKET_COLORS[bucket],
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in rows], rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.text(
            0.5,
            -0.24,
            "Best-effort rules; scheduler time-warp fallback is approximate",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        _save(fig, summary_dir, stem)

    for key, ylabel, stem, title in (
        (
            "phase_count_by_kind",
            "Event count",
            "phase_kind_count_by_experiment",
            "Control/Comm/Compute Counts by Phase and Experiment",
        ),
        (
            "phase_duration_seconds_by_kind",
            "Summed duration (s)",
            "phase_kind_duration_by_experiment",
            "Control/Comm/Compute Durations by Phase and Experiment",
        ),
    ):
        labels = [
            f"{row['label']}\n{phase}"
            for row in rows
            for phase in EVENT_PHASE_ORDER
        ]
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5.5))
        for kind in KIND_ORDER:
            vals = np.array([
                float(row[key].get(phase, {}).get(kind, 0.0))
                for row in rows
                for phase in EVENT_PHASE_ORDER
            ])
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=kind,
                color=KIND_COLORS[kind],
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(
            0.5,
            -0.30,
            "mixed events are credited to both prefill and decode",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        _save(fig, summary_dir, stem)

    for key, ylabel, stem, title in (
        (
            "heuristic_phase_count_by_kind",
            "Event count",
            "heuristic_phase_kind_count_by_experiment",
            "Heuristic Control/Comm/Compute Counts by Phase and Experiment",
        ),
        (
            "heuristic_phase_duration_seconds_by_kind",
            "Summed duration (s)",
            "heuristic_phase_kind_duration_by_experiment",
            "Heuristic Control/Comm/Compute Durations by Phase and Experiment",
        ),
    ):
        if not any(
            row.get(key, {}).get(phase, {})
            for row in rows
            for phase in EVENT_PHASE_ORDER
        ):
            continue
        labels = [
            f"{row['label']}\n{phase}"
            for row in rows
            for phase in EVENT_PHASE_ORDER
        ]
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5.5))
        for kind in KIND_ORDER:
            vals = np.array([
                float(row[key].get(phase, {}).get(kind, 0.0))
                for row in rows
                for phase in EVENT_PHASE_ORDER
            ])
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=kind,
                color=KIND_COLORS[kind],
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(
            0.5,
            -0.30,
            "HEURISTIC: CUPTI trace time linearly warped onto scheduler elapsed; "
            "mixed events credited to both phases",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        _save(fig, summary_dir, stem)

    cluster_rows = [
        row
        for row in rows
        if row["control_comm_burst_cluster"].get("control_events", 0)
    ]
    if cluster_rows:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for row in cluster_rows:
            cluster = row["control_comm_burst_cluster"]
            counts = np.array(
                cluster.get("burst_relative_histogram_mean_density", []),
                dtype=np.float64,
            )
            edges = np.array(
                cluster.get("burst_relative_histogram_bin_edges_ms", []),
                dtype=np.float64,
            )
            if counts.size == 0 or edges.size != counts.size + 1:
                continue
            centers = (edges[:-1] + edges[1:]) / 2.0
            ax.plot(
                centers,
                counts,
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=row["label"],
            )
        ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--")
        ax.set_title("Control Density vs Comm Burst by Experiment (Fig. 4)")
        ax.set_xlabel("Time relative to comm burst start (ms)")
        ax.set_ylabel("Mean control events per burst")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        ax.text(
            0.5,
            -0.20,
            "All phases; 0=burst start; +/-5 ms window",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        _save(
            fig,
            summary_dir,
            "control_comm_burst_density_by_experiment",
        )

        fig, ax = plt.subplots(figsize=(8.5, 5))
        for row in cluster_rows:
            cluster = row["control_comm_burst_cluster"]
            counts = np.array(
                cluster.get("burst_relative_histogram_mean_density", []),
                dtype=np.float64,
            )
            edges = np.array(
                cluster.get("burst_relative_histogram_bin_edges_ms", []),
                dtype=np.float64,
            )
            if counts.size == 0 or edges.size != counts.size + 1:
                continue
            total = counts.sum()
            if total <= 0:
                continue
            centers = (edges[:-1] + edges[1:]) / 2.0
            ax.plot(
                centers,
                counts / total,
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=row["label"],
            )
        ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--")
        ax.set_title("Control Density Shape by Experiment")
        ax.set_xlabel("Time relative to comm burst start (ms)")
        ax.set_ylabel("Fraction of mean control density")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        ax.text(
            0.5,
            -0.20,
            "All phases; 0=burst start; +/-5 ms window",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        _save(
            fig,
            summary_dir,
            "control_comm_burst_density_fraction_by_experiment",
        )

        labels = [row["label"] for row in cluster_rows]
        fractions = [
            float(
                row["control_comm_burst_cluster"].get(
                    "clustered_control_fraction",
                    0.0,
                )
            ) * 100.0
            for row in cluster_rows
        ]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4.5))
        ax.bar(labels, fractions, color="#59a14f")
        ax.set_title("Control Near Comm Burst Start by Experiment")
        ax.set_ylabel("Control events within +/-5 ms of burst start (%)")
        upper = max(5.0, min(100.0, max(fractions) * 1.2 if fractions else 100.0))
        ax.set_ylim(0, upper)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
        for idx, value in enumerate(fractions):
            ax.text(idx, value, f"{value:.1f}%", ha="center", va="bottom")
        ax.text(
            0.5,
            -0.25,
            "All phases; comm burst start anchors",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        _save(
            fig,
            summary_dir,
            "control_comm_burst_cluster_fraction_by_experiment",
        )

    flash_rows = [
        row for row in rows if row["flash_kernel_summary"].get("flash_events", 0)
    ]
    if flash_rows:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for row in flash_rows:
            summary = row["flash_kernel_summary"]
            counts = np.array(
                summary.get(
                    "burst_relative_histogram_mean_density_by_variant",
                    {},
                ).get("run_flash_fwd", []),
                dtype=np.float64,
            )
            edges = np.array(
                summary.get("burst_relative_histogram_bin_edges_ms", []),
                dtype=np.float64,
            )
            if counts.size == 0 or edges.size != counts.size + 1:
                continue
            centers = (edges[:-1] + edges[1:]) / 2.0
            ax.plot(
                centers,
                counts,
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=row["label"],
            )
        ax.axvline(0, color="#222222", linewidth=1.0, linestyle="--")
        ax.set_title("run_flash_fwd Density vs Comm Burst by Experiment")
        ax.set_xlabel("Time relative to comm burst start (ms)")
        ax.set_ylabel("Mean run_flash_fwd events per burst")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        ax.text(
            0.5,
            -0.20,
            "Matched by event name; 0=burst start; +/-5 ms window",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()
        _save(
            fig,
            summary_dir,
            "flash_fwd_kernel_comm_burst_density_by_experiment",
        )

        labels = [row["label"] for row in flash_rows]
        p50 = np.array([
            float(
                row["flash_kernel_summary"]["duration_ms_by_variant"][
                    "run_flash_fwd"
                ]["p50_ms"]
            )
            for row in flash_rows
        ])
        p95 = np.array([
            float(
                row["flash_kernel_summary"]["duration_ms_by_variant"][
                    "run_flash_fwd"
                ]["p95_ms"]
            )
            for row in flash_rows
        ])
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4.8))
        ax.bar(x - width / 2, p50, width, label="p50", color="#4e79a7")
        ax.bar(x + width / 2, p95, width, label="p95", color="#f28e2b")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title("run_flash_fwd Duration by Experiment")
        ax.set_ylabel("Duration (ms)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _save(fig, summary_dir, "flash_fwd_kernel_duration_by_experiment")

    scheduler_rows: list[dict[str, Any]] = []
    for item in index:
        summary_path = Path(item["out_dir"]) / "event_classification_summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text())
        if not data.get("scheduler_iterations"):
            continue
        scheduler_rows.append({
            "label": item["label"],
            "token_totals": data.get("scheduler_token_totals", {}),
            "phase_counts": data.get("scheduler_phase_counts", {}),
            "phase_elapsed_seconds": data.get(
                "scheduler_phase_elapsed_seconds", {}
            ),
        })
    if not scheduler_rows:
        return

    x = np.arange(len(scheduler_rows))
    fig, ax = plt.subplots(figsize=(max(8, len(scheduler_rows) * 1.6), 5))
    bottoms = np.zeros(len(scheduler_rows))
    for key in TOKEN_SPLIT_ORDER:
        vals = np.array([
            float(row["token_totals"].get(key, 0.0))
            for row in scheduler_rows
        ])
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            label=TOKEN_SPLIT_LABELS[key],
            color=TOKEN_SPLIT_COLORS[key],
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in scheduler_rows], rotation=20)
    ax.set_ylabel("Tokens")
    ax.set_title("Scheduler Prefill/Decode Tokens by Experiment")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, summary_dir, "scheduler_prefill_decode_tokens_by_experiment")

    for key, ylabel, stem, title in (
        (
            "phase_counts",
            "Iterations",
            "scheduler_iteration_phases_by_experiment",
            "Scheduler Iteration Phases by Experiment",
        ),
        (
            "phase_elapsed_seconds",
            "Elapsed time (s)",
            "scheduler_phase_elapsed_by_experiment",
            "Scheduler Elapsed Time by Experiment",
        ),
    ):
        fig, ax = plt.subplots(figsize=(max(8, len(scheduler_rows) * 1.6), 5))
        bottoms = np.zeros(len(scheduler_rows))
        for phase in SCHEDULER_PHASE_ORDER:
            vals = np.array([
                float(row[key].get(phase, 0.0))
                for row in scheduler_rows
            ])
            if vals.sum() == 0:
                continue
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=SCHEDULER_PHASE_LABELS[phase],
                color=SCHEDULER_PHASE_COLORS[phase],
                edgecolor="white",
                linewidth=0.4,
            )
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in scheduler_rows], rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        _save(fig, summary_dir, stem)


def evaluate_experiment(path: Path, out_name: str) -> dict[str, Any]:
    label = _experiment_label(path)
    out_dir = path / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(path.glob("ray_worker_nsight/**/*.jsonl"))
    nsys_rep_files = sorted(path.glob("ray_worker_nsight/**/*.nsys-rep"))
    stats = _new_stats(label)
    stats["jsonl_files"] = [str(p) for p in jsonl_files]
    stats["nsys_rep_files"] = [str(p) for p in nsys_rep_files]

    scheduler_iterations, max_num_batched_tokens = collect_scheduler_iterations(
        path, stats
    )
    if scheduler_iterations:
        write_scheduler_summary(stats, scheduler_iterations, out_dir)
        plot_scheduler_all(
            stats,
            scheduler_iterations,
            out_dir,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    if not jsonl_files:
        missing = {
            "label": label,
            "jsonl_files": [],
            "nsys_rep_files": stats["nsys_rep_files"],
            "reason": (
                "No JSONL traces found. Export .nsys-rep with "
                "plotting_tools/export_nsys.sh when nsys is available."
            ),
        }
        (out_dir / "missing_data_report.json").write_text(json.dumps(missing, indent=2))
        write_summary(stats, out_dir)
        return stats

    print(f"Evaluating {path} ({len(jsonl_files)} JSONL traces)", flush=True)
    for jsonl_path in jsonl_files:
        scan_jsonl(jsonl_path, stats, scheduler_iterations)
    write_summary(stats, out_dir)
    plot_all(stats, out_dir)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-name", default="plots_experiment")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    experiments = sorted(
        p for p in results_dir.iterdir()
        if p.is_dir() and p.name.startswith("r32_")
    )
    if not experiments:
        raise SystemExit(f"No experiment directories found under {results_dir}")

    index = []
    for experiment in experiments:
        stats = evaluate_experiment(experiment, args.out_name)
        index.append({
            "experiment": str(experiment),
            "label": stats["label"],
            "events": stats["events"],
            "jsonl_files": len(stats["jsonl_files"]),
            "nsys_rep_files": len(stats["nsys_rep_files"]),
            "unclassified_count": int(sum(stats["unclassified"].values())),
            "scheduler_iterations": stats["scheduler_iterations"],
            "out_dir": str(experiment / args.out_name),
        })

    summary_dir = results_dir / f"{args.out_name}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "index.json").write_text(json.dumps(index, indent=2))
    plot_cross_experiment_summary(index, summary_dir)


if __name__ == "__main__":
    main()
