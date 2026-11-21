"""Produce a deterministic, entirely fabricated SoL-Pi factorial A/B report.

This module cannot launch Hermes, invoke providers, run tasks or grade code.
"""
from __future__ import annotations

import argparse
import json

ARMS = {
    "base": (False, False),
    "observation_pack": (True, False),
    "action_fusion": (False, True),
    "both": (True, True),
}
TASKS = ("synthetic/parser-whitespace", "synthetic/format-empty-list", "synthetic/sort-tie-order")
REPLICATES = 4


def policy(observation_pack: bool, action_fusion: bool) -> dict:
    return {
        "version": 1, "observationPack": observation_pack, "actionFusion": action_fusion,
        "evidencePreservingReducer": False, "onlineContextCompact": False,
    }


def fake_job(task: str, task_index: int, arm: str, replicate: int, order: int) -> dict:
    observation, fusion = ARMS[arm]
    turns = 3 + task_index + replicate % 2 - int(fusion)
    return {
        "task_id": task, "replicate": replicate, "arm": arm, "execution_order": order,
        "seed": 20261120 + task_index * 100 + replicate,
        "argv_preview": [
            "hermes", "chat", "-q", "<reviewed task prompt>", "--oneshot",
            "--format", "stream-json", "--toolsets", "terminal,file,sol-pi",
            "--max-turns", "30", "--run-budget", "600",
        ],
        "policy": policy(observation, fusion), "metric_source": "fake",
        # Fixed illustrative variation, not modeled treatment effects.
        "passed": (task_index + replicate + tuple(ARMS).index(arm)) % 5 != 0,
        "wall_ms": 600 + turns * 170 + replicate * 31,
        "user_turns": 1, "model_response_turns": turns,
        "requests": [
            {
                "request_index": i, "status": "completed",
                "usage": {
                    "input_tokens": 1200 + i * (110 if observation else 200),
                    "cache_read_tokens": 0 if i == 1 else 500 + 50 * i,
                    "cache_write_tokens": 600 if i == 1 else 0,
                    "output_tokens": 90 + i * 13,
                },
            }
            for i in range(1, turns + 1)
        ],
    }


def build_report() -> dict:
    jobs = []
    arms = tuple(ARMS)
    for task_index, task in enumerate(TASKS):
        for replicate in range(1, REPLICATES + 1):
            rotation = (task_index + replicate - 1) % len(arms)
            ordered = arms[rotation:] + arms[:rotation]
            jobs.extend(fake_job(task, task_index, arm, replicate, order)
                        for order, arm in enumerate(ordered, 1))
    summary = {}
    for arm in ARMS:
        selected = [job for job in jobs if job["arm"] == arm]
        passed = sum(job["passed"] for job in selected)
        summary[arm] = {"jobs": len(selected), "passed": passed, "pass_rate": passed / len(selected)}
    return {
        "schema_version": 1,
        "mode": "dry_run",
        "live_execution_enabled": False,
        "usage_source": "fake",
        "benchmark_target": "SWE-bench Verified reviewed subset (not selected in dry run)",
        "noninferiority": {
            "primary_contrast": ["both", "base"], "margin_percentage_points": -3,
            "confidence_level": 0.95, "interval_side": "one_sided_lower",
            "resampling_unit": "task", "decision": None, "lower_bound_percentage_points": None,
        },
        "jobs": jobs,
        "summary": summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="emit fake report without running any command")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("--dry-run is required; live execution is not implemented")
    print(json.dumps(build_report(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
