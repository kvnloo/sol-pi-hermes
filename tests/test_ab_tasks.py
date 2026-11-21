"""The A/B eval tasks are solvable, deterministic, and graded on behavior, not on agent-edited files."""
import tempfile
import unittest
from pathlib import Path

from evals import tasks as T


def solve(task_id: str, wd: Path) -> None:
    def sub(name, old, new):
        p = wd / name
        text = p.read_text()
        assert old in text, (name, old)
        p.write_text(text.replace(old, new))

    if task_id == "fix_parser":
        sub("inventory.py", 'split(",")', 'split(";")')
    elif task_id == "multi_fix":
        sub("textstats.py", "text.split(' ')", "text.split()")
        sub("report.py", "most_common(n + 1)][1:]", "most_common(n)]")
    elif task_id == "config_audit":
        gold = sorted(s["name"] for s in T._services() if not s["tls"] and s["port"] < 1024)
        (wd / "answer.txt").write_text("\n".join(gold) + "\n")
    elif task_id == "log_trace":
        (wd / "answer.txt").write_text(T._log_world()[2] + "\n")
    elif task_id == "chained_edits":
        sub("ledger.py", "whole, _, frac = text.partition('.')", "whole, _, frac = text.replace(',', '').partition('.')")
        sub("ledger.py", "sum(c for a, k, c in rows if a == account)",
            "sum(c if k == 'credit' else -c for a, k, c in rows if a == account)")
        (wd / "ledger.py").write_text((wd / "ledger.py").read_text() + (
            "\n\ndef top_accounts(rows, n):\n"
            "    names = sorted({a for a, _, _ in rows}, key=lambda a: -balance(rows, a))\n"
            "    return names[:n]\n\n\n"
            "def export_csv(rows, path):\n"
            "    with open(path, 'w') as f:\n"
            "        f.write('account,kind,cents\\n' + ''.join(f'{a},{k},{c}\\n' for a, k, c in rows))\n"))
    else:
        raise AssertionError(task_id)


class AbTasks(unittest.TestCase):
    def test_each_task_fails_unsolved_and_passes_solved(self):
        for task in T.TASKS:
            with self.subTest(task.id), tempfile.TemporaryDirectory() as d:
                wd = Path(d)
                task.build(wd)
                self.assertTrue(any(p.stat().st_size > 10 * 1024 for p in wd.iterdir()),
                                "each task needs a file above ObservationPack's 10 KiB threshold")
                self.assertFalse(task.score(wd, "")["pass"])
                solve(task.id, wd)
                self.assertTrue(task.score(wd, "")["pass"], task.score(wd, ""))

    def test_tests_are_graded_from_pristine_copies(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            T.TASKS[0].build(wd)
            (wd / "test_inventory.py").write_text("import unittest\n")  # agent deletes the tests
            self.assertFalse(T.TASKS[0].score(wd, "")["pass"])

    def test_generation_is_byte_stable(self):
        outs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                for task in T.TASKS:
                    (Path(d) / task.id).mkdir()
                    task.build(Path(d) / task.id)
                outs.append({p.relative_to(d).as_posix(): p.read_bytes() for p in Path(d).rglob("*") if p.is_file()})
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
