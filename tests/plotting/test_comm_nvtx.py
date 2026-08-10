# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from plotting_tools.comm_nvtx import (
    finalize_comm_nvtx_records,
    parse_comm_nvtx_label,
    parse_iter_nvtx_label,
)


def _with_interval(
    record: dict,
    *,
    ts: int = 10,
    end: int = 20,
) -> dict:
    return {
        **record,
        "ts": ts,
        "end": end,
        "dur_us": end - ts,
        "scope": record.get("scope", "comm"),
    }


def test_parse_canonical_iteration_label() -> None:
    iteration = parse_iter_nvtx_label(
        "iter|id=37|phase=prefill|rank=3|pp=0|tp=3|ctx=128|gen=0|"
        "reqs=4"
    )

    assert iteration is not None
    assert iteration["iter_id"] == 37
    assert iteration["phase"] == "prefill"
    assert iteration["rank"] == 3
    assert iteration["ctx_tokens"] == 128
    assert iteration["gen_tokens"] == 0
    assert iteration["reqs"] == 4


def test_parse_and_finalize_canonical_comm_label() -> None:
    comm = parse_comm_nvtx_label(
        "comm|iter=37|op=all_reduce|phase=prefill|shape=128x4096|"
        "bytes=1048576|rank=3|pp=0|tp=3"
    )
    iteration = parse_iter_nvtx_label(
        "iter|id=37|phase=prefill|rank=3|pp=0|tp=3|ctx=128|gen=0|"
        "reqs=4"
    )

    assert comm is not None
    assert iteration is not None
    assert comm["iter_id"] == 37
    assert comm["op"] == "all_reduce"
    assert comm["phase"] == "prefill"
    assert comm["logical_tensor_bytes"] == 1048576

    finalized, stats = finalize_comm_nvtx_records(
        [_with_interval(comm)],
        [{**iteration, "ts": 0, "end": 100}],
    )

    assert stats["phases_assigned_from_iteration"] == 0
    assert finalized[0]["phase"] == "prefill"
    assert finalized[0]["phase_source"] == "comm_nvtx_label"
    assert finalized[0]["logical_tensor_bytes"] == 1048576


def test_parse_supported_canonical_comm_ops_and_phases() -> None:
    cases = [
        ("all_reduce", "decode", 38, 512),
        ("all_gather", "mixed", 39, 1024),
        ("reduce_scatter", "prefill", 40, 2048),
        ("all_reduce", "unknown", -1, 4096),
    ]

    for op, phase, iter_id, nbytes in cases:
        comm = parse_comm_nvtx_label(
            f"comm|iter={iter_id}|op={op}|phase={phase}|shape=16x16|"
            f"bytes={nbytes}|rank=1|pp=0|tp=1"
        )

        assert comm is not None
        assert comm["iter_id"] == iter_id
        assert comm["op"] == op
        assert comm["phase"] == phase
        assert comm["logical_tensor_bytes"] == nbytes


def test_finalize_stamps_non_unknown_pipe_phase_sources() -> None:
    records = []
    for phase in ("decode", "mixed"):
        comm = parse_comm_nvtx_label(
            f"comm|iter=4|op=all_gather|phase={phase}|shape=8x8|"
            "bytes=128|rank=0|pp=0|tp=0"
        )
        assert comm is not None
        records.append(_with_interval(comm))

    finalized, _stats = finalize_comm_nvtx_records(records, [])

    assert [record["phase"] for record in finalized] == ["decode", "mixed"]
    assert [record["phase_source"] for record in finalized] == [
        "comm_nvtx_label",
        "comm_nvtx_label",
    ]


def test_unknown_iter_minus_one_remains_unknown_after_finalize() -> None:
    comm = parse_comm_nvtx_label(
        "comm|iter=-1|op=reduce_scatter|phase=unknown|shape=8x8|"
        "bytes=128|rank=0|pp=0|tp=0"
    )
    iteration = parse_iter_nvtx_label(
        "iter|id=4|phase=decode|rank=0|pp=0|tp=0|ctx=0|gen=4|reqs=4"
    )

    assert comm is not None
    assert iteration is not None

    finalized, stats = finalize_comm_nvtx_records(
        [_with_interval(comm)],
        [{**iteration, "ts": 0, "end": 100}],
    )

    assert stats["phases_assigned_from_iteration"] == 0
    assert finalized[0]["iter_id"] == -1
    assert finalized[0]["phase"] == "unknown"
    assert finalized[0].get("phase_source") is None


def test_iter_id_fallback_is_rank_aware() -> None:
    records = [
        _with_interval({
            "op": "all_reduce",
            "phase": "unknown",
            "iter_id": 7,
            "shape": "1",
            "logical_tensor_bytes": 4,
            "rank": 3,
        }),
        _with_interval({
            "op": "all_reduce",
            "phase": "unknown",
            "iter_id": 7,
            "shape": "1",
            "logical_tensor_bytes": 4,
            "rank": 4,
        }),
    ]
    iteration_ranges = [
        {
            "name": "prefill",
            "phase": "prefill",
            "iter_id": 7,
            "rank": 3,
            "ts": 0,
            "end": 100,
        },
        {
            "name": "decode",
            "phase": "decode",
            "iter_id": 7,
            "rank": 4,
            "ts": 0,
            "end": 100,
        },
    ]

    finalized, stats = finalize_comm_nvtx_records(records, iteration_ranges)

    assert stats["phases_assigned_from_iteration"] == 2
    assert [record["phase"] for record in finalized] == ["prefill", "decode"]
    assert [record["phase_source"] for record in finalized] == [
        "iter_id",
        "iter_id",
    ]


def test_overlap_fallback_requires_matching_known_rank() -> None:
    record = _with_interval({
        "op": "all_reduce",
        "phase": "unknown",
        "iter_id": None,
        "shape": "1",
        "logical_tensor_bytes": 4,
        "rank": 3,
    })
    iteration_ranges = [
        {
            "name": "decode",
            "phase": "decode",
            "iter_id": 2,
            "rank": 4,
            "ts": 0,
            "end": 50,
        },
        {
            "name": "prefill",
            "phase": "prefill",
            "iter_id": 2,
            "rank": 3,
            "ts": 0,
            "end": 100,
        },
    ]

    finalized, stats = finalize_comm_nvtx_records([record], iteration_ranges)

    assert stats["phases_assigned_from_iteration"] == 1
    assert finalized[0]["phase"] == "prefill"
    assert finalized[0]["phase_source"] == "iteration_nvtx_overlap"
