#!/usr/bin/env python3
"""Download ShareGPT Vicuna data and export vLLM-ready prompts.

Example:
  uv run python datasets/prepare_sharegpt_vicuna_prompts.py \
    --max-rows 100000 \
    --raw-out datasets/sharegpt_sample.json \
    --prompts-out datasets/sharegpt_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASET = "Aeala/ShareGPT_Vicuna_unfiltered"


def _looks_human_role(role: str) -> bool:
    role_norm = role.strip().lower()
    return role_norm in {"human", "user", "prompt"}


def _extract_prompt_from_row(row: dict[str, Any]) -> str | None:
    """Extract a single prompt string from one ShareGPT row.

    Priority:
      1) First human/user turn from conversation-style rows.
      2) Common instruction-style text fields.
    """
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", "")))
            if _looks_human_role(role):
                value = turn.get("value", turn.get("content", ""))
                text = str(value).strip()
                if text:
                    return text

    messages = row.get("messages")
    if isinstance(messages, list):
        for turn in messages:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", turn.get("from", "")))
            if _looks_human_role(role):
                value = turn.get("content", turn.get("value", ""))
                text = str(value).strip()
                if text:
                    return text

    for key in ("prompt", "instruction", "input", "question", "text"):
        value = row.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument(
        "--raw-out",
        default="datasets/sharegpt_sample.json",
        help="Path to write raw sampled rows as JSON.",
    )
    parser.add_argument(
        "--prompts-out",
        default="datasets/sharegpt_prompts.jsonl",
        help="Path to write extracted prompts as JSONL.",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    n = min(args.max_rows, len(ds))
    rows = [ds[i] for i in range(n)]

    raw_out = Path(args.raw_out)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with raw_out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)

    prompts_out = Path(args.prompts_out)
    prompts_out.parent.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with prompts_out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            prompt = _extract_prompt_from_row(row)
            if not prompt:
                continue
            record = {
                "id": i,
                "prompt": prompt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            extracted += 1

    print(
        f"Loaded {n} rows from {args.dataset}:{args.split}. "
        f"Wrote {extracted} prompts to {prompts_out}."
    )


if __name__ == "__main__":
    main()
