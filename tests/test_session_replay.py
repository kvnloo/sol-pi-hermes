import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay_hermes_session.py"


class SessionReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sol-pi-replay-test-")
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "snapshot.sqlite"
        with sqlite3.connect(self.db) as db:
            db.executescript("""
                CREATE TABLE sessions(
                    id TEXT PRIMARY KEY, parent_session_id TEXT,
                    api_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER,
                    cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER
                );
                CREATE TABLE messages(
                    id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT,
                    tool_calls TEXT, tool_name TEXT, timestamp REAL, active INTEGER,
                    compacted INTEGER, api_content TEXT
                );
                INSERT INTO sessions VALUES ('primary',NULL,9,1234,99,102,21,3),
                    ('second','primary',0,0,0,0,0,0), ('pending',NULL,NULL,NULL,NULL,NULL,NULL,NULL);
            """)
            calls = [{"id":"call","type":"function","function":{"name":"read_file","arguments":"{}"}}]
            large = "café saved result α\n" * 2200
            rows = [
                (1,"primary","user","display only",None,None,None,100,1,0,"durable user sidecar"),
                (2,"primary","assistant",None,None,json.dumps(calls),None,99,1,0,None),
                (3,"primary","tool",large,"call",None,"read_file",98,1,0,None),
                (4,"primary","assistant","first",None,None,None,97,1,0,None),
                (5,"primary","tool","compacted text" * 4000,"expired",None,"read_file",96,0,1,None),
                (6,"primary","user","continue",None,None,None,95,1,0,None),
                (7,"primary","assistant","second",None,None,None,94,1,0,None),
                (8,"primary","user","continue",None,None,None,93,1,0,None),
                (9,"primary","assistant","third",None,None,None,92,1,0,None),
                (10,"primary","tool","\0json:"+json.dumps([{"type":"image_url","image_url":{"url":"data:image/png;base64," + "A" * 20000}}]),"image",None,"computer_use",91,1,0,None),
                (11,"primary","assistant","fourth",None,None,None,90,1,0,None),
                (12,"second","user","same observation",None,None,None,200,1,0,None),
                (13,"second","assistant",None,None,json.dumps(calls),None,199,1,0,None),
                (14,"second","tool",large,"call",None,"read_file",198,1,0,None),
                (15,"second","assistant","one send only",None,None,None,197,1,0,None),
                (16,"pending","user","not proof of a call",None,None,None,196,1,0,None),
            ]
            db.executemany("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(self.db), *args],
            capture_output=True, text=True, check=False,
        )

    def test_projection_metrics_are_per_session_read_only_and_recall_pages_are_exact(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        p = self.run_cli("primary", "second", "pending")
        self.assertEqual(p.returncode, 0, p.stderr)
        report = json.loads(p.stdout)
        first, second, pending = report["sessions"]
        self.assertEqual([r["assistant_row_id"] for r in first["boundaries"]], [2, 4, 7, 9, 11])
        self.assertEqual([r["placeholder_count"] for r in first["boundaries"]], [0, 0, 0, 1, 1])
        self.assertGreater(first["totals"]["saved_message_json_bytes"], 0)
        self.assertGreater(first["totals"]["saved_estimated_content_text_tokens"], 0)
        self.assertEqual(first["observed_session_usage"]["input_tokens"], 1234)
        self.assertEqual(first["observed_session_usage"]["api_call_count"], 9)
        self.assertEqual(first["inferred_request_opportunities"], 5)
        sidecar_message = [{"role": "user", "content": "durable user sidecar"}]
        self.assertEqual(first["boundaries"][0]["before_message_json_bytes"],
                         len(json.dumps(sidecar_message, separators=(",", ":")).encode()))
        self.assertIsNone(first["actual_projection_token_delta"])
        self.assertEqual(len(first["recall"]), 1)
        self.assertGreater(first["recall"][0]["page_count"], 1)
        self.assertTrue(first["recall"][0]["byte_exact"])
        self.assertGreater(first["recall"][0]["largest_page_json_bytes"], 0)
        self.assertLessEqual(first["recall"][0]["largest_page_json_bytes"], 16 * 1024)
        self.assertEqual([r["placeholder_count"] for r in second["boundaries"]], [0, 0])
        self.assertEqual(second["recall"], [])
        self.assertEqual(pending["boundaries"], [])
        self.assertIsNone(pending["observed_session_usage"]["input_tokens"])
        for private in ("café saved result", "durable user sidecar", "image/png", "compacted text"):
            self.assertNotIn(private, p.stdout)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)

    def test_missing_session_and_unknown_schema_fail_without_opening_writable(self):
        unknown = self.run_cli("no-such-session")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("not found", unknown.stderr)
        self.assertEqual(unknown.stdout, "")
        with sqlite3.connect(self.db) as db:
            db.execute("DROP TABLE messages")
        incompatible = self.run_cli("primary")
        self.assertEqual(incompatible.returncode, 2)
        self.assertIn("unsupported", incompatible.stderr)
        self.assertEqual(incompatible.stdout, "")
