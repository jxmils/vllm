#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEMA_VERSION = "1.0"
_TRACER_LOCK = threading.Lock()
_TRACER: "MoEStatsTracer | None" = None


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class MoEStatsTracer:
    def __init__(self, base_dir: str | None = None):
        try:
            config = get_current_vllm_config()
        except AssertionError:
            config = None
        parallel_config = config.parallel_config if config is not None else None
        model_name = (
            config.model_config.model
            if config is not None and config.model_config is not None
            else None
        )
        default_root = Path(__file__).resolve().parents[2] / "traces"
        root_dir = Path(base_dir) if base_dir else default_root
        rank = int(os.environ.get("RANK", "0"))
        run_id = (
            (config.instance_id if config is not None else "")
            or os.environ.get("VLLM_INSTANCE_ID", "")
            or _now_utc_compact()
        )
        run_name = f"moe_stats_{run_id}"
        self.run_dir = root_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / f"router_events_rank_{rank}.jsonl"
        self.selection_path = (
            self.run_dir / f"selected_experts_rank_{rank}.jsonl"
        )
        self._fp = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._selection_fp = self.selection_path.open(
            "a", encoding="utf-8", buffering=1
        )
        self.gpu_timings_path = self.run_dir / f"gpu_op_timings_rank_{rank}.jsonl"
        self._gpu_fp = self.gpu_timings_path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._event_count = 0
        self._selection_index = 0
        self._decode_entries_written = 0
        self._max_decode_entries = 128
        self._active_selection_entry: dict[str, Any] | None = None
        self._last_layer_id: int | None = None
        self._write_metadata(
            rank=rank,
            model_name=model_name,
            parallel_config=parallel_config,
        )
        atexit.register(self.flush_and_close)

    def _write_metadata(
        self,
        rank: int,
        model_name: str | None,
        parallel_config: Any | None,
    ) -> None:
        metadata_path = self.run_dir / "metadata.json"
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at_unix_ns": time.time_ns(),
            "run_dir": str(self.run_dir),
            "model_name": model_name,
            "pid": os.getpid(),
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "dp_size": (
                parallel_config.data_parallel_size
                if parallel_config is not None
                else None
            ),
            "tp_size": (
                parallel_config.tensor_parallel_size
                if parallel_config is not None
                else None
            ),
            "ep_enabled": (
                parallel_config.enable_expert_parallel
                if parallel_config is not None
                else None
            ),
            "all2all_backend": (
                parallel_config.all2all_backend
                if parallel_config is not None
                else None
            ),
        }
        # Best effort; avoid clobbering from other ranks.
        try:
            if not metadata_path.exists():
                metadata_path.write_text(
                    json.dumps(metadata, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
        except Exception:
            logger.exception("Failed to write MoE stats metadata.")

    def write_event(self, event: dict[str, Any]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, **event}
        try:
            line = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            logger.exception("Failed to serialize MoE stats event.")
            return
        with self._lock:
            self._fp.write(line + "\n")
            self._event_count += 1

    def write_gpu_timing_event(self, event: dict[str, Any]) -> None:
        payload = {"schema_version": SCHEMA_VERSION, **event}
        try:
            line = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            logger.exception("Failed to serialize GPU timing event.")
            return
        with self._lock:
            self._gpu_fp.write(line + "\n")

    def add_layer_selection(
        self,
        *,
        layer_id: int | None,
        request_phase: str | None,
        selected_experts: list[list[int]],
    ) -> None:
        # Unknown layer index cannot be grouped meaningfully.
        if layer_id is None:
            return
        phase = request_phase or "unknown"
        with self._lock:
            # Start a new output-token element on layer id wraparound.
            if self._active_selection_entry is None or (
                self._last_layer_id is not None and layer_id <= self._last_layer_id
            ):
                self._flush_active_selection_locked()
                self._active_selection_entry = {
                    "output_token_index": self._selection_index,
                    "request_phase": phase,
                    "layers": {},
                }
                self._selection_index += 1
            assert self._active_selection_entry is not None
            self._active_selection_entry["request_phase"] = phase
            self._active_selection_entry["layers"][str(layer_id)] = selected_experts
            self._last_layer_id = layer_id

    def _flush_active_selection_locked(self) -> None:
        if self._active_selection_entry is None:
            return
        phase = self._active_selection_entry.get("request_phase")
        if phase == "decode" and self._decode_entries_written >= self._max_decode_entries:
            self._active_selection_entry = None
            self._last_layer_id = None
            return
        line = json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                **self._active_selection_entry,
            },
            sort_keys=True,
            default=str,
        )
        self._selection_fp.write(line + "\n")
        if phase == "decode":
            self._decode_entries_written += 1
        self._active_selection_entry = None
        self._last_layer_id = None

    def flush_and_close(self) -> None:
        with self._lock:
            self._flush_active_selection_locked()
            try:
                self._fp.flush()
            except Exception:
                logger.exception("Failed to flush MoE stats trace file.")
            try:
                self._selection_fp.flush()
            except Exception:
                logger.exception("Failed to flush selected experts trace file.")
            try:
                self._gpu_fp.flush()
            except Exception:
                logger.exception("Failed to flush GPU timings trace file.")
            try:
                self._fp.close()
            except Exception:
                logger.exception("Failed to close MoE stats trace file.")
            try:
                self._selection_fp.close()
            except Exception:
                logger.exception("Failed to close selected experts trace file.")
            try:
                self._gpu_fp.close()
            except Exception:
                logger.exception("Failed to close GPU timings trace file.")


def get_moe_stats_tracer() -> MoEStatsTracer | None:
    global _TRACER
    with _TRACER_LOCK:
        if _TRACER is not None:
            return _TRACER
        try:
            config = get_current_vllm_config()
        except AssertionError:
            return None
        if (
            config is None
            or config.observability_config is None
            or not config.observability_config.track_moe_stats
        ):
            return None
        _TRACER = MoEStatsTracer(config.observability_config.track_moe_stats_dir)
        return _TRACER
