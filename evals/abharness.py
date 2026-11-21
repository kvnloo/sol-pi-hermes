"""Frozen A/B harness: plain Hermes vs Hermes + this repo's plugin, same tasks, same model.

Controller (any python3 >= 3.10, stdlib only):

    python3 evals/abharness.py run --tasks evals.tasks --out evals/results/<run> \
        --hermes-source ~/.hermes/hermes-agent --hermes-python ~/.hermes/hermes-agent/venv/bin/python \
        --work /tmp/ab-work --repeats 3
    python3 evals/abharness.py report evals/results/<run>

Each row gets a fresh HERMES_HOME (generated config only: no credentials, memories, sessions or
skills are copied) and a fresh working directory built by the task. The child process calls
Hermes's own ``run_agent.AIAgent.run_conversation``, so plugin discovery, tool assembly,
middleware and dispatch are Hermes-owned. Token and call counts are Hermes's session counters,
which sum the provider's reported usage. The task module defines the arms; the harness never
changes between arms.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha256_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(p.relative_to(REPO).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def git_sha(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------- child (inside Hermes venv)

def child(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        from run_agent import AIAgent
        agent = AIAgent(base_url=spec["base_url"], api_key=os.environ[spec["key_env"]],
                        provider=spec["provider"], requested_provider=spec["provider"],
                        api_mode="chat_completions", model=spec["model"],
                        max_iterations=spec["max_iterations"], max_tokens=spec["max_tokens"],
                        enabled_toolsets=spec["toolsets"], quiet_mode=True, save_trajectories=False,
                        skip_context_files=True, skip_memory=True, skip_background_review=True,
                        session_db=None, fallback_model={}, session_id=spec["session_id"],
                        platform="cli", cwd=os.getcwd())
        try:
            result = agent.run_conversation(user_message=spec["prompt"])
            Path(spec["transcript"]).write_text(json.dumps(result, ensure_ascii=False, default=str))
            calls = [c.get("function", {}).get("name", c.get("name"))
                     for m in result.get("messages", []) for c in (m.get("tool_calls") or [])]
            summary = {"answer": result.get("final_response") or "", "completed": result.get("completed"),
                       "failed": bool(result.get("failed")), "error": str(result.get("error") or ""),
                       "turn_exit_reason": str(result.get("turn_exit_reason") or ""),
                       "tool_calls": calls, "tools_exposed": sorted(agent.valid_tool_names),
                       "input_tokens": agent.session_input_tokens,
                       "output_tokens": agent.session_output_tokens,
                       "api_calls": agent.session_api_calls}
        finally:
            agent.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


# ---------------------------------------------------------------- controller

def fresh_home(home: Path, arm: dict, args) -> None:
    home.mkdir(parents=True)
    provider = {"base_url": args.base_url, "api_key_env": args.key_env, "key_env": args.key_env,
                "api_mode": "chat_completions"}
    aux = {"provider": args.provider, "base_url": args.base_url, "api_key_env": args.key_env,
           "model": args.sub_model, "fallback_models": []}
    config = {
        "_config_version": 12,
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
        "model": {"provider": args.provider, "default": args.model, "max_tokens": args.max_tokens},
        "providers": {args.provider: provider},
        "auxiliary": {"transient_retries": 1, "compression": dict(aux), **{t: dict(aux) for t in arm.get("aux_tasks", [])},
                      "background_review": {"enabled": False}},
        "plugins": {"enabled": [arm["plugin_name"]] if arm.get("plugin_dir") else [],
                    "entries": arm.get("plugin_entries", {})},
        "tools": {"tool_search": {"enabled": "off"}},
        "model_catalog": {"enabled": False},
        "agent": {"max_turns": args.max_iterations, "api_max_retries": 3, "auto_recovery_cycles": 0},
        "terminal": {"backend": "local"},
        "approvals": {"mode": "off"},
    }
    # YAML is a JSON superset, so the controller needs no YAML dependency.
    (home / "config.yaml").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    if arm.get("plugin_dir"):
        shutil.copytree(REPO / arm["plugin_dir"], home / "plugins" / arm["plugin_name"],
                        ignore=shutil.ignore_patterns("__pycache__", ".git", "tests", "evals", "bench",
                                                      "*.pyc", ".github", "schema"))
    for name, text in arm.get("home_files", {}).items():
        (home / name).write_text(text)


def run_child(cmd, cwd, env, timeout):
    start = time.monotonic()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            p.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
        out.seek(0), err.seek(0)
        return p.returncode, out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace"), \
            timed_out, time.monotonic() - start


def plugin_log_stats(home: Path) -> dict:
    """Event counts from plugin JSONL logs in this row's home, plus RLM sub-call token sums.

    Hermes's session counters cover root requests only; RLM sub-calls go through the auxiliary
    lane and are recorded once each, as `subcall` events, in the plugin's trajectory log.
    """
    events, sub_in, sub_out, sub_missing = {}, 0, 0, 0
    for log in sorted(home.rglob("*.jsonl")):
        for line in log.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(e.get("event"))
            events[name] = events.get(name, 0) + 1
            if name == "subcall":
                if isinstance(e.get("prompt_tokens"), int) and isinstance(e.get("completion_tokens"), int):
                    sub_in += e["prompt_tokens"]
                    sub_out += e["completion_tokens"]
                else:
                    sub_missing += 1
    return {"plugin_events": events, "sub_input_tokens": sub_in, "sub_output_tokens": sub_out,
            "sub_usage_missing": sub_missing}


def run(args) -> int:
    sys.path.insert(0, str(REPO))
    tasks_mod = importlib.import_module(args.tasks)
    tasks, arms = tasks_mod.TASKS, tasks_mod.ARMS
    if args.task_ids:
        tasks = [t for t in tasks if t.id in set(args.task_ids.split(","))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(l)["key"] for l in rows_path.read_text().splitlines() if l.strip()}
    plugin_files = [p for p in REPO.rglob("*") if p.is_file() and (p.suffix in {".py", ".yaml"})
                    and not any(part in {".git", "tests", "evals", "bench", "__pycache__"} for part in p.relative_to(REPO).parts)]
    protocol = {"harness_sha": sha256_files([Path(__file__)]),
                "tasks_sha": sha256_files([Path(tasks_mod.__file__)]),
                "plugin_sha": sha256_files(plugin_files), "repo_sha": git_sha(REPO),
                "hermes_sha": git_sha(Path(args.hermes_source)), "model": args.model,
                "sub_model": args.sub_model, "base_url": args.base_url,
                "max_iterations": args.max_iterations, "max_tokens": args.max_tokens}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1, sort_keys=True) + "\n")
    work = Path(args.work)
    arm_names = list(arms)
    for rep in range(args.repeats):
        for ti, task in enumerate(tasks):
            k = (rep + ti) % len(arm_names)  # rotate which arm goes first
            for arm_name in arm_names[k:] + arm_names[:k]:
                key = f"{task.id}|{arm_name}|{rep}|{protocol['harness_sha']}|{protocol['tasks_sha']}|{protocol['plugin_sha']}|{args.model}"
                if key in done:
                    continue
                run_id = f"{task.id}-{arm_name}-{rep}-{int(time.time())}"
                home, wd = work / run_id / "home", work / run_id / "cwd"
                arm = arms[arm_name]
                fresh_home(home, arm, args)
                wd.mkdir(parents=True)
                task.build(wd)
                spec = {"base_url": args.base_url, "key_env": args.key_env, "provider": args.provider,
                        "model": args.model, "max_iterations": args.max_iterations,
                        "max_tokens": args.max_tokens, "toolsets": arm["toolsets"],
                        "session_id": "ab-" + hashlib.sha256(run_id.encode()).hexdigest()[:16],
                        "prompt": arm.get("prompt_template", "{prompt}").format(prompt=task.prompt),
                        "transcript": str(work / run_id / "transcript.json")}
                (work / run_id / "spec.json").write_text(json.dumps(spec))
                env = {"PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(home),
                       "PYTHONPATH": str(Path(args.hermes_source).resolve()), "LANG": "C.UTF-8",
                       args.key_env: os.environ.get(args.key_env, "anonymous"),
                       "HERMES_DISABLE_TELEMETRY": "1", "TERM": "dumb"}
                cmd = [args.hermes_python, "-B", str(Path(__file__).resolve()), "child", str(work / run_id / "spec.json")]
                code, stdout, stderr, timed_out, wall = run_child(cmd, wd, env, args.timeout)
                (work / run_id / "stderr.txt").write_text(stderr)
                try:
                    summary = json.loads(stdout.strip().splitlines()[-1])
                except (IndexError, json.JSONDecodeError):
                    summary = {"answer": "", "error": f"child exit {code}; no summary", "tool_calls": []}
                grade = task.score(wd, summary.get("answer", ""))
                row = {"key": key, "task": task.id, "arm": arm_name, "repeat": rep, "run_id": run_id,
                       "pass": bool(grade["pass"]), "grade": grade, "timed_out": timed_out,
                       "exit_code": code, "wall_s": round(wall, 1), **protocol, **plugin_log_stats(home),
                       **{k2: summary.get(k2) for k2 in ("input_tokens", "output_tokens", "api_calls",
                                                         "tool_calls", "tools_exposed", "error",
                                                         "turn_exit_reason", "completed")},
                       "answer": (summary.get("answer") or "")[:2000]}
                with rows_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"{run_id}: pass={row['pass']} in_tok={row['input_tokens']} calls={row['api_calls']} "
                      f"wall={row['wall_s']}s err={str(row['error'])[:80]!r}", flush=True)
    return 0


# ---------------------------------------------------------------- report

def report(args) -> int:
    out = Path(args.out)
    rows = [json.loads(l) for l in (out / "rows.jsonl").read_text().splitlines() if l.strip()]
    protocol = json.loads((out / "protocol.json").read_text())
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    infra = lambda r: r["timed_out"] or (r["api_calls"] in (None, 0))  # no model answer at all
    tot = lambda r: None if r["input_tokens"] is None else r["input_tokens"] + r.get("sub_input_tokens", 0)

    def agg(rs):
        ok = [r for r in rs if not infra(r)]
        tok = [tot(r) for r in ok if tot(r) is not None]
        return {"n": len(rs), "infra_fail": len(rs) - len(ok), "pass": sum(r["pass"] for r in rs),
                "in_tok_mean": round(statistics.mean(tok)) if tok else None,
                "in_tok_median": round(statistics.median(tok)) if tok else None,
                "api_calls_mean": round(statistics.mean([r["api_calls"] for r in ok]), 1) if ok else None,
                "wall_mean": round(statistics.mean([r["wall_s"] for r in ok]), 1) if ok else None,
                "events": {k: sum(r.get("plugin_events", {}).get(k, 0) for r in rs)
                           for k in sorted({k for r in rs for k in r.get("plugin_events", {})})}}

    lines = ["# Results", "", "Generated by `python3 evals/abharness.py report " + args.out + "` from `rows.jsonl`.", "",
             "Protocol: " + ", ".join(f"`{k}={v}`" for k, v in sorted(protocol.items())), "",
             "Input tokens are provider-reported prompt tokens: root requests (Hermes session counters) plus "
             "plugin sub-calls (`subcall` events in the plugin log; RLM only).", "",
             "## By arm", "", "| arm | runs | pass | infra fail | mean input tokens | median input tokens | mean API calls | mean wall s | plugin log events |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = agg([r for r in rows if r["arm"] == a])
        lines.append(f"| {a} | {s['n']} | {s['pass']}/{s['n']} | {s['infra_fail']} | {s['in_tok_mean']} | "
                     f"{s['in_tok_median']} | {s['api_calls_mean']} | {s['wall_mean']} | "
                     + (", ".join(f"{k} {v}" for k, v in s["events"].items()) or "-") + " |")
    lines += ["", "## By task", "", "| task | arm | pass | mean input tokens | mean API calls |", "|---|---|---|---|---|"]
    for t in tasks:
        for a in arms:
            s = agg([r for r in rows if r["arm"] == a and r["task"] == t])
            lines.append(f"| {t} | {a} | {s['pass']}/{s['n']} | {s['in_tok_mean']} | {s['api_calls_mean']} |")
    base = arms[0] if args.baseline is None else args.baseline
    pairs = {}
    for r in rows:
        pairs.setdefault((r["task"], r["repeat"]), {})[r["arm"]] = r
    lines += ["", f"## Paired against `{base}` (same task and repeat, both arms answered)", "",
              "| arm | pairs | pass flips (+/-) | input-token ratio, median | input-token ratio, per-task medians |", "|---|---|---|---|---|"]
    for a in arms:
        if a == base:
            continue
        ps = [(p[base], p[a]) for p in pairs.values() if base in p and a in p and not infra(p[base]) and not infra(p[a])]
        ratios = [tot(y) / tot(x) for x, y in ps if tot(x) and tot(y) is not None]
        per_task = {t: [tot(y) / tot(x) for x, y in ps if x["task"] == t and tot(x) and tot(y) is not None] for t in tasks}
        lines.append(f"| {a} | {len(ps)} | +{sum(y['pass'] and not x['pass'] for x, y in ps)}/-{sum(x['pass'] and not y['pass'] for x, y in ps)} | "
                     f"{round(statistics.median(ratios), 3) if ratios else None} | "
                     + ", ".join(f"{t} {round(statistics.median(v), 2)}" for t, v in per_task.items() if v) + " |")
    lines += ["", "## Rows", "", "| task | arm | rep | pass | input tok | API calls | tools used | note |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["task"], r["repeat"], r["arm"])):
        used = {}
        for c in r["tool_calls"] or []:
            used[c] = used.get(c, 0) + 1
        note = "timeout" if r["timed_out"] else (str(r["error"])[:60] or r["grade"].get("detail", "")[:60])
        lines.append(f"| {r['task']} | {r['arm']} | {r['repeat']} | {'y' if r['pass'] else 'n'} | {tot(r)} | "
                     f"{r['api_calls']} | {', '.join(f'{k}x{v}' for k, v in sorted(used.items()))} | {note.replace('|', '/')} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("child")
    c.add_argument("spec")
    r = sub.add_parser("run")
    r.add_argument("--tasks", required=True, help="module defining TASKS and ARMS, e.g. evals.tasks")
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True, help="scratch dir for per-row homes and working dirs")
    r.add_argument("--hermes-source", required=True)
    r.add_argument("--hermes-python", required=True)
    r.add_argument("--base-url", default="https://api.kilo.ai/api/gateway")
    r.add_argument("--provider", default="kilo")
    r.add_argument("--key-env", default="AB_PROVIDER_KEY")
    r.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b:free")
    r.add_argument("--sub-model", default="nvidia/nemotron-3.5-lightning:free")
    r.add_argument("--max-iterations", type=int, default=24)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--timeout", type=int, default=600)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--task-ids")
    s = sub.add_parser("report")
    s.add_argument("out")
    s.add_argument("--baseline")
    args = p.parse_args(argv)
    return child(args.spec) if args.cmd == "child" else run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    sys.exit(main())
