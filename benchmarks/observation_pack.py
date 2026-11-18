"""Synthetic request-projection measurement, never provider billing telemetry.

Run from the checkout: python -m benchmarks.observation_pack
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Callable

from sol_pi_hermes.observation_pack import create_observation, project_request, recall


def wire(value: object) -> str:
    """A fixed measurement serialization, not a provider's transport framing."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def measure(messages: list[dict], requests: int = 4,
            counter: Callable[[str], int] | None = None,
            counter_name: str | None = None) -> dict:
    if requests < 1:
        raise ValueError("requests must be positive")
    if counter_name is None:
        counter_name = "unnamed_custom_counter" if counter else "ceil_unicode_chars_div_4_estimate"
    counter = counter or (lambda text: math.ceil(len(text) / 4))
    original = copy.deepcopy(messages)
    source = {"messages": messages}
    rows = []
    with tempfile.TemporaryDirectory(prefix="sol-pi-measure-") as tmp:
        root = Path(tmp)
        sent = {}
        last = source
        for index in range(requests):
            projection = project_request(source, root, sent)
            last = projection["request"] if projection else source
            before, after = wire(source), wire(last)
            rows.append({"request": index + 1, "control_bytes": len(before.encode()),
                         "projected_bytes": len(after.encode()), "control_units": counter(before),
                         "projected_units": counter(after)})
            if source["messages"] != original:
                raise AssertionError("projection mutated the source transcript")

        # Account for one complete recall of each unique packed row. This is
        # response payload overhead only, not future requests carrying recall.
        recalled, recall_bytes, recall_units = set(), 0, 0
        recall_roundtrip = True
        for before, after in zip(original, last["messages"]):
            if before == after:
                continue
            observation = create_observation(before, root)
            if observation is None or observation.id in recalled:
                continue
            recalled.add(observation.id)
            offset, pieces = 0, []
            while True:
                result = json.loads(recall(root, observation.id, offset))
                if "error" in result:
                    raise AssertionError(result["error"])
                serialized = wire(result)
                recall_bytes += len(serialized.encode())
                recall_units += counter(serialized)
                pieces.append(result["content"].split("\n", 2)[2])
                if result["eof"]:
                    break
                if result["next_offset"] <= offset:
                    raise AssertionError("recall made no progress")
                offset = result["next_offset"]
            recall_roundtrip &= "".join(pieces) == observation.text
        control_bytes = sum(row["control_bytes"] for row in rows)
        projected_bytes = sum(row["projected_bytes"] for row in rows)
        control_units = sum(row["control_units"] for row in rows)
        projected_units = sum(row["projected_units"] for row in rows)
        return {"counter": counter_name, "provider_billed_tokens": None, "requests": rows,
                "control_bytes": control_bytes, "projected_bytes": projected_bytes,
                "saved_bytes_before_recall": control_bytes - projected_bytes,
                "control_units": control_units, "projected_units": projected_units,
                "saved_units_before_recall": control_units - projected_units,
                "recall_response_bytes": recall_bytes, "recall_response_units": recall_units,
                "net_saved_units_after_one_full_recall_response": control_units - projected_units - recall_units,
                "recalled_observations": len(recalled), "recall_roundtrip": recall_roundtrip,
                "transcript_unchanged": messages == original}


def fixtures() -> dict[str, list[dict]]:
    def row(text, **extra):
        return {"role": "tool", "name": "terminal", "tool_call_id": "synthetic", "content": text, **extra}
    return {"large_ascii": [row("synthetic result line\n" * 1000)],
            "large_unicode": [row("測定🙂 synthetic\n" * 1000)],
            "small_control": [row("tiny")],
            "error_control": [row("synthetic error\n" * 1000, isError=True)],
            "mixed_image_control": [row([{"type": "text", "text": "x" * 16000},
                                          {"type": "image_url", "image_url": {"url": "synthetic://image"}}])]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--encoding", help="Optional installed tiktoken encoding; never billed tokens")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    counter, name = None, None
    if args.encoding:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding(args.encoding)
        except (ImportError, ValueError) as exc:
            parser.error(f"cannot load requested tokenizer: {exc}")
        counter = lambda text: len(encoding.encode(text, disallowed_special=()))
        name = f"tiktoken:{args.encoding}:serialized_request_not_billing"
    print(json.dumps({key: measure(value, args.requests, counter, name)
                      for key, value in fixtures().items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
