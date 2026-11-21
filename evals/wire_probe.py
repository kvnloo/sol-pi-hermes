"""Does sol-pi's llm_request projection reach the wire? Fake OpenAI server records each request.

Usage: python3 evals/wire_probe.py . <hermes source> <hermes python>   (no network, no key)
Scripted model: read big.txt, then 5 small terminal calls, then answer. Prints per-request body bytes.
"""
import argparse, json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

repo, hsrc, hpy = map(Path, sys.argv[1:4])
sys.path.insert(0, str(repo))
from evals import abharness as H, tasks as T  # noqa: E402

SCRIPT = [("read_file", {"path": "big.txt"})] + [("terminal", {"command": f"echo step{i}"}) for i in range(5)]
LOG = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        req = json.loads(body)
        if "messages" not in req:
            print("   non-chat request", self.path, sorted(req)[:12], flush=True)
            self.send_response(404)
            self.end_headers()
            return
        n = sum(1 for m in req["messages"] if m["role"] == "assistant")
        tool_text = sum(len(m.get("content") or "") if isinstance(m.get("content"), str) else len(json.dumps(m.get("content")))
                        for m in req["messages"] if m["role"] == "tool")
        LOG.append({"bytes": len(body), "tool_content_chars": tool_text, "tools": len(req.get("tools") or [])})
        if n < len(SCRIPT):
            name, args = SCRIPT[n]
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        else:
            msg, finish = {"role": "assistant", "content": "done"}, "stop"
        usage = {"prompt_tokens": len(body) // 4, "completion_tokens": 5, "total_tokens": len(body) // 4 + 5}
        if req.get("stream"):
            delta = {"role": "assistant", "content": msg.get("content")}
            if msg.get("tool_calls"):
                delta["tool_calls"] = [dict(tc, index=0) for tc in msg["tool_calls"]]
            chunks = [{"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "fake:free",
                       "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                      {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "fake:free",
                       "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}]
            out = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"
            ctype = "text/event-stream"
        else:
            out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "fake:free",
                              "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


srv = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/v1"
work = Path(os.environ.get("PROBE_WORK", "/tmp/wire-probe")) / str(int(time.time()))
for arm_name in ("plain", "sol-pi"):
    LOG.clear()
    arm = T.ARMS[arm_name]
    home, wd = work / arm_name / "home", work / arm_name / "cwd"
    args = argparse.Namespace(base_url=base, key_env="AB_PROVIDER_KEY", provider="ab-lane", model="fake:free",
                              sub_model="fake:free", max_tokens=256, max_iterations=12)
    H.fresh_home(home, arm, args)
    wd.mkdir(parents=True)
    (wd / "big.txt").write_text("".join(f"line {i:05d} " + "x" * 60 + "\n" for i in range(600)))  # ~44 KB
    spec = {"base_url": base, "key_env": "AB_PROVIDER_KEY", "provider": "ab-lane", "model": "fake:free",
            "max_iterations": 12, "max_tokens": 256, "toolsets": arm["toolsets"], "session_id": "probe-" + arm_name,
            "prompt": "Read big.txt then run the steps.", "transcript": str(work / arm_name / "t.json")}
    (work / arm_name / "spec.json").write_text(json.dumps(spec))
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "HERMES_HOME": str(home), "PYTHONPATH": str(hsrc),
           "AB_PROVIDER_KEY": "x", "LANG": "C.UTF-8", "TERM": "dumb"}
    p = subprocess.run([str(hpy), "-B", str(repo / "evals/abharness.py"), "child", str(work / arm_name / "spec.json")],
                       cwd=wd, env=env, capture_output=True, text=True, timeout=300)
    summary = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"err": p.stderr[-800:]}
    print(f"== {arm_name}: input_tokens={summary.get('input_tokens')} api_calls={summary.get('api_calls')} err={summary.get('err', '')[:300]}")
    for i, e in enumerate(LOG, 1):
        print(f"   request {i}: body {e['bytes']} B, tool-result chars {e['tool_content_chars']}, tools {e['tools']}")
    ledgers = list(home.rglob("ledger.jsonl"))
    if ledgers:
        for l in ledgers[0].read_text().splitlines():
            e = json.loads(l)
            print("   ledger:", e["event"], "request", e.get("request"), "bytes", e.get("originalBytes"))
