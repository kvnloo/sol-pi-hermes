from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from sol_pi_hermes.action_fusion import THEN_RUN_SKIPPED, THEN_RUN_SUCCEEDED, execute_mutation_then_run
from sol_pi_hermes.config import load_sol_pi_config
from sol_pi_hermes.epr import LIKELY_SECRET, quote_verify, should_reduce, transform_if_reduced
from sol_pi_hermes.observation_pack import (
    FULL_SENDS,
    THRESHOLD_BYTES,
    create_observation,
    placeholder_for,
    project_messages,
    recall,
    wrap_select_context,
)
from sol_pi_hermes.occ import OnlineCompactGate
from sol_pi_hermes.plugin import register


def _big_tool(text: str, tool_call_id: str = "tc1", name: str = "terminal") -> dict:
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": tool_call_id,
        "content": text,
    }


class ConfigTests(unittest.TestCase):
    def test_defaults_all_off(self) -> None:
        cfg = load_sol_pi_config(raw={})
        self.assertFalse(cfg.observation_pack)
        self.assertFalse(cfg.action_fusion)
        self.assertFalse(cfg.evidence_preserving_reducer)
        self.assertFalse(cfg.online_context_compact)

    def test_unknown_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_sol_pi_config(raw={"version": 1, "nope": True})


class ObservationPackTests(unittest.TestCase):
    def test_small_results_are_not_packed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            msg = _big_tool("tiny")
            self.assertIsNone(create_observation(msg, root))

    def test_projection_does_not_mutate_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "L\n" * (THRESHOLD_BYTES // 2 + 50)
            history = [
                {"role": "user", "content": "go"},
                _big_tool(body),
            ]
            sent: dict[str, int] = {}
            first = project_messages(history, root, sent)
            self.assertEqual(history[1]["content"], body)
            self.assertEqual(first[1]["content"], body)
            second = project_messages(history, root, sent)
            third = project_messages(history, root, sent)
            self.assertEqual(FULL_SENDS, 2)
            self.assertIn("obs_", placeholder_for(create_observation(history[1], root)))
            self.assertIn("[large tool result replaced", third[1]["content"])
            self.assertEqual(history[1]["content"], body)
            self.assertNotIn("[large tool result replaced", second[1]["content"])

    def test_reducer_receipt_is_not_packed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "sol_pi_evidence_receipt_v1\n" + ("x" * (THRESHOLD_BYTES + 10))
            self.assertIsNone(create_observation(_big_tool(body), root))

    def test_obs_recall_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "line\n" * (THRESHOLD_BYTES // 5 + 20)
            history = [_big_tool(body)]
            sent: dict[str, int] = {}
            project_messages(history, root, sent)
            project_messages(history, root, sent)
            packed = project_messages(history, root, sent)
            obs_id = packed[0]["content"].split("id: ", 1)[1].split("\n", 1)[0]
            payload = json.loads(recall(root, obs_id, 0))
            self.assertIn("content", payload)
            self.assertIn("next_offset", payload)

    def test_wrap_select_context_does_not_replace_compress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = SimpleNamespace(compress=lambda messages: messages, should_compress=lambda *a, **k: False)

            def original_select(messages, **kwargs):
                return None

            engine.select_context = original_select
            body = "Z\n" * (THRESHOLD_BYTES // 2 + 40)
            msgs = [_big_tool(body)]
            sent: dict[str, int] = {}
            wrap_select_context(engine, root, sent)
            engine.select_context(msgs)
            engine.select_context(msgs)
            out = engine.select_context(msgs)
            self.assertTrue(callable(engine.compress))
            self.assertTrue(any("obs_" in str(m.get("content")) for m in out))
            self.assertEqual(msgs[0]["content"], body)


class OccTests(unittest.TestCase):
    def test_no_compact_without_agent_settled(self) -> None:
        gate = OnlineCompactGate(enabled=True)
        gate.on_plan_boundary()
        self.assertFalse(gate.should_compact())
        self.assertEqual(gate.last_reason, "waiting_for_agent_settled")
        idle = OnlineCompactGate(enabled=True)
        self.assertFalse(idle.on_agent_settled())

    def test_compact_only_after_settle_and_boundary(self) -> None:
        gate = OnlineCompactGate(enabled=True)
        gate.on_plan_boundary()
        self.assertTrue(gate.on_agent_settled())

    def test_disabled_never_compacts(self) -> None:
        gate = OnlineCompactGate(enabled=False)
        gate.on_plan_boundary()
        self.assertFalse(gate.on_agent_settled())


class EprTests(unittest.TestCase):
    def test_secrets_are_not_reduced(self) -> None:
        body = "api_key=sekret\n" + ("e" * 5000)
        self.assertTrue(LIKELY_SECRET.search(body))
        self.assertFalse(should_reduce(body, "cargo test"))

    def test_quote_verify_fail_open(self) -> None:
        self.assertFalse(quote_verify(["missing"], "log"))
        self.assertTrue(quote_verify(["error: boom"], "error: boom\nrest"))
        self.assertIsNone(
            transform_if_reduced(
                tool_name="terminal",
                args={"command": "cargo test"},
                result="x" * 5000,
                reducer=None,
            )
        )


class FusionTests(unittest.TestCase):
    def test_then_run_skipped_on_mutate_failure(self) -> None:
        def mutate() -> str:
            raise RuntimeError("write failed")

        out = json.loads(
            execute_mutation_then_run(
                mutate=mutate,
                absolute_path=Path("/tmp/nope"),
                then_run={"command": "true"},
                run_command=lambda *_: "ok",
            )
        )
        self.assertIn(THEN_RUN_SKIPPED, out["error"])

    def test_then_run_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.py"
            path.write_text("print(1)\n", encoding="utf-8")

            def mutate() -> str:
                return "wrote"

            out = json.loads(
                execute_mutation_then_run(
                    mutate=mutate,
                    absolute_path=path,
                    then_run={"command": "python f.py"},
                    run_command=lambda cmd, _t: f"ran {cmd}",
                )
            )
            self.assertEqual(out["then_run"], THEN_RUN_SUCCEEDED)


class RegisterTests(unittest.TestCase):
    def test_register_does_not_register_context_engine(self) -> None:
        ctx = MagicMock()
        ctx._manager = SimpleNamespace(_context_engine=None, _cli_ref=None)
        from sol_pi_hermes.config import SolPiConfig

        register(ctx, SolPiConfig(observation_pack=True))
        ctx.register_context_engine.assert_not_called()
        names = [c.kwargs["name"] for c in ctx.register_tool.call_args_list]
        self.assertIn("obs_recall", names)
        hooks = [c.args[0] for c in ctx.register_hook.call_args_list]
        self.assertIn("agent_settled", hooks)
        self.assertIn("transform_tool_result", hooks)


if __name__ == "__main__":
    unittest.main()
