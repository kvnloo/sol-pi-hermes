from __future__ import annotations

import unittest

from sol_pi_hermes.observation_pack import hash_text
from sol_pi_hermes.epr import MIN_BYTES, RECEIPT_PREFIX, apply_receipt, should_reduce, transform_if_reduced

BIG = ("error: boom\n" + "padding line\n" * 600)  # well over MIN_BYTES, receipt far smaller


def _run(result=BIG, *, tool="terminal", args=None, reducer=None):
    return transform_if_reduced(
        tool_name=tool,
        args={"command": "pytest -q"} if args is None else args,
        result=result,
        reducer=reducer,
    )


def _ok(**_):
    return {"quotes": ["error: boom"], "notes": "n"}


class EprGatePins(unittest.TestCase):
    def test_min_bytes_boundary(self) -> None:
        self.assertFalse(should_reduce("a" * (MIN_BYTES - 1), "pytest"))
        self.assertTrue(should_reduce("a" * MIN_BYTES, "pytest"))

    def test_command_matching_is_case_insensitive(self) -> None:
        self.assertTrue(should_reduce("a" * MIN_BYTES, "PYTEST -q"))

    def test_non_diagnostic_command_is_never_reduced(self) -> None:
        self.assertFalse(should_reduce("a" * MIN_BYTES, "echo hi"))

    def test_python3_module_runners_are_diagnostic(self) -> None:
        for cmd in ("python3 -m unittest", "python3 -m py_compile x.py"):
            self.assertTrue(should_reduce("a" * MIN_BYTES, cmd), cmd)

    def test_secret_gate_blocks_reduction(self) -> None:
        self.assertFalse(should_reduce("api_key=abc\n" + "a" * MIN_BYTES, "pytest"))

    def test_every_diagnostic_runner_is_matched(self) -> None:
        for cmd in ("lake build", "lean f.lean", "coq", "cargo test", "zig build", "ctest", "cmake --build b",
                    "ninja", "make", "npm test", "pnpm test", "yarn test", "go test ./...", "bazel test //x"):
            self.assertTrue(should_reduce("a" * MIN_BYTES, cmd), cmd)

    def test_every_secret_marker_blocks_reduction(self) -> None:
        for marker in ("authorization: x", "bearer=x", "access-token=x", "secret=x", "api_key: x", "api_key" + "z" * 32 + "=x"):
            self.assertFalse(should_reduce(marker + "\n" + "a" * MIN_BYTES, "pytest"), marker)
        self.assertTrue(should_reduce("api_key" + "z" * 33 + "=x\n" + "a" * MIN_BYTES, "pytest"))

    def test_min_bytes_counts_bytes_not_characters(self) -> None:
        self.assertTrue(should_reduce("é" * (MIN_BYTES // 2), "pytest"))

    def test_tool_names_and_cmd_alias(self) -> None:
        for tool in ("terminal", "bash", "execute"):
            self.assertIsNotNone(_run(tool=tool, reducer=_ok), tool)
        self.assertIsNone(_run(tool="read_file", reducer=_ok))
        self.assertIsNotNone(_run(args={"cmd": "pytest -q"}, reducer=_ok))
        self.assertIsNone(_run(args={}, reducer=_ok))  # no command at all is never reduced


class EprFailOpen(unittest.TestCase):
    def test_no_reducer_keeps_original(self) -> None:
        self.assertIsNone(_run(reducer=None))

    def test_reducer_exception_keeps_original(self) -> None:
        def boom(**_):
            raise RuntimeError("reducer down")

        self.assertIsNone(_run(reducer=boom))

    def test_non_dict_reducer_output_keeps_original(self) -> None:
        self.assertIsNone(_run(reducer=lambda **_: ["error: boom"]))
        self.assertIsNone(_run(reducer=lambda **_: "error: boom"))

    def test_non_list_quotes_keep_original(self) -> None:
        self.assertIsNone(_run(reducer=lambda **_: {"quotes": "error: boom", "notes": ""}))

    def test_unverifiable_quote_keeps_original(self) -> None:
        self.assertIsNone(_run(reducer=lambda **_: {"quotes": ["not in the log"], "notes": ""}))

    def test_one_unverifiable_quote_among_valid_ones_keeps_original(self) -> None:
        self.assertIsNone(_run(reducer=lambda **_: {"quotes": ["error: boom", "not in the log"], "notes": ""}))

    def test_receipt_not_smaller_keeps_original(self) -> None:
        # Receipt strictly larger than a body that is just over MIN_BYTES.
        body = "error: boom\n" + "a" * MIN_BYTES
        self.assertIsNone(_run(result=body, reducer=lambda **_: {"quotes": [body], "notes": ""}))

    def test_receipt_equal_size_keeps_original(self) -> None:
        # Build a body whose receipt has exactly the same byte length.
        quote = "q" * 10
        base = len(apply_receipt("pytest -q", "0" * 64, [quote], "").encode())
        body = quote + "\n" + "a" * (MIN_BYTES)
        pad = len(body.encode()) - base
        notes = "n" * pad
        receipt = apply_receipt("pytest -q", hash_text(body), [quote], notes)
        self.assertEqual(len(receipt.encode()), len(body.encode()))
        self.assertIsNone(_run(result=body, reducer=lambda **_: {"quotes": [quote], "notes": notes}))


class EprReceipt(unittest.TestCase):
    def test_receipt_format(self) -> None:
        receipt = _run(reducer=lambda **_: {"quotes": ["error: boom", "padding line"], "notes": None})
        lines = receipt.split("\n")
        self.assertEqual(lines[0], RECEIPT_PREFIX)
        self.assertEqual(lines[1], "command: pytest -q")
        self.assertIn("- error: boom", lines)
        self.assertIn("- padding line", lines)
        self.assertEqual(lines[-1], "")  # missing notes -> empty, never the string "None"
        self.assertEqual(lines[2], f"source_sha256: {hash_text(BIG)}")
        self.assertNotIn("None", receipt)
        self.assertEqual(lines[3:7], ["quotes:", "- error: boom", "- padding line", "notes:"])
        self.assertEqual(_run(reducer=lambda **_: {"quotes": ["error: boom"], "notes": "why"}).split("\n")[-2:], ["notes:", "why"])
