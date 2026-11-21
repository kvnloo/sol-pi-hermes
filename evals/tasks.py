"""A/B tasks for sol-pi: plain Hermes vs Hermes + sol-pi (ObservationPack + Action Fusion).

Every task is generated deterministically into a fresh working directory, needs at least one
large (>10 KiB) file the agent will usually read, and keeps working for several turns after
that read. That is the workload ObservationPack targets: a large tool result that would be
resent on every later request. Grading is deterministic and never trusts files the agent
could have edited: test tasks rerun pristine copies of the original tests.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REVISION = "solpi-ab-v1"


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    build: Callable[[Path], None]
    score: Callable[[Path, str], dict]


def _rng(name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{REVISION}|{name}".encode()).hexdigest()[:16], 16))


def _run_pristine_tests(wd: Path, modules: list[str], tests: dict[str, str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        for m in modules:
            if not (wd / m).is_file():
                return {"pass": False, "detail": f"{m} missing"}
            shutil.copyfile(wd / m, Path(tmp) / m)
        for name, text in tests.items():
            (Path(tmp) / name).write_text(text)
        p = subprocess.run([sys.executable, "-I", "-m", "unittest", "-q"], cwd=tmp,
                           capture_output=True, text=True, timeout=60)
        return {"pass": p.returncode == 0, "detail": (p.stderr.strip().splitlines() or [""])[-1][:120]}


# ---------------------------------------------------------------- fix_parser

def _catalog(r: random.Random, n: int) -> dict:
    words = ["alpha", "bravo", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "juniper"]
    kinds = ["widget", "bracket", "spindle", "gasket", "flange", "coupler", "bearing", "sprocket"]
    return {f"SKU-{i:04d}": (f"{r.choice(kinds).title()} {r.choice(words)} {i}", round(r.uniform(1, 250), 2))
            for i in range(1, n + 1)}


INVENTORY_TEST = '''import unittest
from inventory import parse_line, order_total, CATALOG

class T(unittest.TestCase):
    def test_parse_line(self):
        self.assertEqual(parse_line("SKU-0007;3;2026-11-01\\n"), ("SKU-0007", 3, "2026-11-01"))

    def test_order_total(self):
        lines = ["SKU-0001;2;2026-11-01", "SKU-0420;1;2026-11-02"]
        want = round(CATALOG["SKU-0001"][1] * 2 + CATALOG["SKU-0420"][1], 2)
        self.assertEqual(order_total(lines), want)

if __name__ == "__main__":
    unittest.main()
'''


def build_fix_parser(wd: Path) -> None:
    cat = _catalog(_rng("fix_parser"), 480)
    body = ['"""Order parsing for the warehouse export. Lines look like SKU;QTY;DATE."""', "",
            "def parse_line(line):",
            '    """Return (sku, qty, date) for one export line."""',
            '    sku, qty, date = line.strip().split(",")',
            "    return sku, int(qty), date", "", "",
            "def order_total(lines):",
            "    total = 0.0",
            "    for line in lines:",
            "        sku, qty, _ = parse_line(line)",
            "        total += CATALOG[sku][1] * qty",
            "    return round(total, 2)", "", "",
            "CATALOG = {"]
    body += [f"    {k!r}: ({v[0]!r}, {v[1]!r})," for k, v in cat.items()]
    body += ["}", ""]
    (wd / "inventory.py").write_text("\n".join(body))
    (wd / "test_inventory.py").write_text(INVENTORY_TEST)


def score_fix_parser(wd: Path, answer: str) -> dict:
    return _run_pristine_tests(wd, ["inventory.py"], {"test_inventory.py": INVENTORY_TEST})


# ---------------------------------------------------------------- multi_fix

TEXTSTATS_TEST = '''import unittest
from textstats import word_count
from report import top_words

class T(unittest.TestCase):
    def test_word_count_whitespace(self):
        self.assertEqual(word_count("a  b\\tc\\nd"), 4)

    def test_top_words(self):
        text = "kiwi apple kiwi pear apple kiwi plum"
        self.assertEqual(top_words(text, 2), ["kiwi", "apple"])

if __name__ == "__main__":
    unittest.main()
'''


def build_multi_fix(wd: Path) -> None:
    r = _rng("multi_fix")
    stop = sorted({"".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(r.randint(3, 9))) for _ in range(1500)})
    (wd / "textstats.py").write_text(
        '"""Text statistics used by report.py."""\n\n'
        "def word_count(text):\n"
        '    """Number of words, where words are separated by any whitespace."""\n'
        "    return len([w for w in text.split(' ') if w])\n\n\n"
        "def words(text):\n"
        "    return [w.lower() for w in text.split() if w.lower() not in STOPWORDS]\n\n\n"
        "STOPWORDS = frozenset([\n" + "".join(f"    {w!r},\n" for w in stop) + "])\n")
    weights = {f"w{i:04d}": round(r.uniform(0.1, 3.0), 3) for i in range(700)}
    (wd / "report.py").write_text(
        '"""Frequency report."""\nfrom collections import Counter\nfrom textstats import words\n\n\n'
        "def top_words(text, n):\n"
        '    """The n most frequent non-stopwords, most frequent first (ties: first seen)."""\n'
        "    counts = Counter(words(text))\n"
        "    return [w for w, _ in counts.most_common(n + 1)][1:]\n\n\n"
        "# Legacy scoring weights, unused by top_words.\nWEIGHTS = {\n"
        + "".join(f"    {k!r}: {v!r},\n" for k, v in weights.items()) + "}\n")
    (wd / "test_textstats.py").write_text(TEXTSTATS_TEST)


def score_multi_fix(wd: Path, answer: str) -> dict:
    return _run_pristine_tests(wd, ["textstats.py", "report.py"], {"test_textstats.py": TEXTSTATS_TEST})


# ---------------------------------------------------------------- config_audit

def _services():
    r = _rng("config_audit")
    out = []
    for i in range(220):
        name = f"svc-{r.choice(['auth', 'billing', 'search', 'media', 'ledger', 'notify', 'edge', 'batch'])}-{i:03d}"
        port = r.choice([80, 443, 389, 636, 25, 587, 993, 8080, 8443, 9000, 5432, 6379, 3000, 22, 53, 110])
        tls = r.random() < 0.7
        out.append({"name": name, "port": port, "tls": tls, "owner": f"team-{r.randint(1, 30)}",
                    "replicas": r.randint(1, 12), "region": r.choice(["us-east", "us-west", "eu", "ap"]),
                    "healthcheck": f"/health/{name}", "env": {"LOG_LEVEL": r.choice(["info", "debug", "warn"])}})
    return out


def build_config_audit(wd: Path) -> None:
    lines = ["# Service registry. One entry per service.", "services:"]
    for s in _services():
        lines += [f"  - name: {s['name']}", f"    port: {s['port']}", f"    tls: {str(s['tls']).lower()}",
                  f"    owner: {s['owner']}", f"    replicas: {s['replicas']}", f"    region: {s['region']}",
                  f"    healthcheck: {s['healthcheck']}", "    env:", f"      LOG_LEVEL: {s['env']['LOG_LEVEL']}"]
    (wd / "services.yaml").write_text("\n".join(lines) + "\n")


def score_config_audit(wd: Path, answer: str) -> dict:
    gold = sorted(s["name"] for s in _services() if not s["tls"] and s["port"] < 1024)
    p = wd / "answer.txt"
    got = [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.is_file() else None
    return {"pass": got == gold, "detail": f"gold {len(gold)} names, got {None if got is None else len(got)}"}


# ---------------------------------------------------------------- log_trace

def _log_world():
    r = _rng("log_trace")
    users = [(f"u{i:04d}", f"{r.choice(['ana', 'bo', 'cy', 'dee', 'eli', 'fay'])}.{i:04d}@example.test",
              r.choice(["free", "pro", "team"])) for i in range(600)]
    target_req, target_user = "R-5521", users[457][0]
    lines, t = [], 0
    for i in range(2600):
        t += r.randint(1, 4)
        req = f"R-{r.randint(1000, 9999)}"
        if req == target_req:
            req = "R-1000"
        u = r.choice(users)[0]
        level = "WARN" if r.random() < 0.05 else "INFO"
        code = f" code=E{r.choice([4300, 4301, 4310, 4313, 4321])}" if level == "WARN" else ""
        lines.append(f"2026-11-03T{10 + t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}Z {level} req={req} user={u} "
                     f"path=/api/{r.choice(['orders', 'cart', 'search', 'login'])} latency={r.randint(3, 900)}ms{code}")
        if i == 1777:
            lines.append(f"2026-11-03T10:58:01Z INFO req={target_req} user={target_user} path=/api/checkout latency=41ms")
        if i == 1790:
            lines.append(f"2026-11-03T10:58:40Z ERROR req={target_req} path=/api/checkout code=E4312 msg=payment capture failed")
    return lines, users, users[457][1]


def build_log_trace(wd: Path) -> None:
    lines, users, _ = _log_world()
    (wd / "app.log").write_text("\n".join(lines) + "\n")
    (wd / "users.csv").write_text("user_id,email,plan\n" + "".join(f"{u},{e},{p}\n" for u, e, p in users))


def score_log_trace(wd: Path, answer: str) -> dict:
    gold = _log_world()[2]
    p = wd / "answer.txt"
    got = p.read_text().strip() if p.is_file() else None
    return {"pass": got == gold, "detail": f"got {got!r}"[:120]}


# ---------------------------------------------------------------- chained_edits (a long session)

LEDGER_TEST = '''import os, tempfile, unittest
import ledger

class T(unittest.TestCase):
    def test_balance_counts_debits_negative(self):
        self.assertEqual(ledger.balance([("A1", "credit", 500), ("A1", "debit", 120)], "A1"), 380)

    def test_parse_amount_thousands(self):
        self.assertEqual(ledger.parse_amount("1,234.50"), 123450)

    def test_top_accounts(self):
        rows = [("A1", "credit", 5), ("B2", "credit", 9), ("C3", "credit", 7), ("B2", "debit", 1)]
        self.assertEqual(ledger.top_accounts(rows, 2), ["B2", "C3"])

    def test_export_csv(self):
        rows = [("A1", "credit", 5), ("B2", "debit", 3)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.csv")
            ledger.export_csv(rows, p)
            self.assertEqual(open(p).read(), "account,kind,cents\\nA1,credit,5\\nB2,debit,3\\n")

    def test_known_accounts_are_real(self):
        self.assertIn("ACC-0399", ledger.KNOWN_ACCOUNTS)
        self.assertEqual(ledger.KNOWN_ACCOUNTS["ACC-0399"][1], ledger.owner_of("ACC-0399"))

if __name__ == "__main__":
    unittest.main()
'''


def build_chained_edits(wd: Path) -> None:
    r = _rng("chained_edits")
    accts = {f"ACC-{i:04d}": (f"{r.choice(['Ops', 'Sales', 'R&D', 'Legal', 'Payroll'])} {i}",
                              f"{r.choice(['ana', 'bo', 'cy', 'dee'])}{i}") for i in range(400)}
    (wd / "ledger.py").write_text(
        '"""Account ledger. Amounts are integer cents. Rows are (account, kind, cents)."""\n\n'
        "def parse_amount(text):\n"
        '    """\'12.50\' -> 1250. TODO: accept thousands separators like \'1,234.50\'."""\n'
        "    whole, _, frac = text.partition('.')\n"
        "    return int(whole) * 100 + int((frac + '00')[:2])\n\n\n"
        "def balance(rows, account):\n"
        '    """Credits add, debits subtract."""\n'
        "    return sum(c for a, k, c in rows if a == account)\n\n\n"
        "def owner_of(account):\n"
        "    return KNOWN_ACCOUNTS[account][1]\n\n\n"
        "# TODO: top_accounts(rows, n): the n accounts with the highest balance, highest first.\n"
        "# TODO: export_csv(rows, path): write a header line account,kind,cents then one line per row.\n\n\n"
        "KNOWN_ACCOUNTS = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in accts.items()) + "}\n")
    (wd / "test_ledger.py").write_text(LEDGER_TEST)


def score_chained_edits(wd: Path, answer: str) -> dict:
    return _run_pristine_tests(wd, ["ledger.py"], {"test_ledger.py": LEDGER_TEST})


TASKS = [
    Task("fix_parser", "The unit tests in this directory fail. Find and fix the bug in inventory.py "
         "(do not edit the tests) and confirm `python3 -m unittest` passes.", build_fix_parser, score_fix_parser),
    Task("multi_fix", "`python3 -m unittest` fails in this directory. Fix the bugs in the library modules "
         "(do not edit the tests) and confirm the tests pass.", build_multi_fix, score_multi_fix),
    Task("config_audit", "services.yaml is our service registry. Write answer.txt listing every service whose "
         "tls is false and whose port is below 1024: names only, sorted ascending, one per line.",
         build_config_audit, score_config_audit),
    Task("log_trace", "app.log has exactly one ERROR with code E4312. Find which user made that request "
         "(other log lines for the same req id name the user), look up the user's email in users.csv, "
         "and write only the email to answer.txt.", build_log_trace, score_log_trace),
    Task("chained_edits", "Work through ledger.py: fix balance(), make parse_amount() accept thousands "
         "separators, and implement the two TODO functions. Do one change at a time and run "
         "`python3 -m unittest` after each change. Do not edit the tests.", build_chained_edits, score_chained_edits),
]

SOLPI_JSON = json.dumps({"version": 1, "observationPack": True, "actionFusion": True})
ARMS = {
    "plain": {"toolsets": ["file", "terminal"]},
    "sol-pi": {"toolsets": ["file", "terminal", "sol-pi"], "plugin_dir": ".", "plugin_name": "sol-pi",
               "home_files": {"sol-pi.json": SOLPI_JSON}},
}
