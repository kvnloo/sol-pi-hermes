"""Shared sol-pi.json policy v1: Hermes loader accepts the canonical file.

Does not remint ObservationPack. Constants are imported from the existing
implementation and checked against schema/$defs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.config import load_sol_pi_config
from sol_pi_hermes.observation_pack import FULL_SENDS, THRESHOLD_BYTES

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schema" / "sol-pi.policy.v1.schema.json"
INSTANCE_PATH = ROOT / "schema" / "sol-pi.policy.v1.json"
OPERATOR_KEYS = {
    "version",
    "actionFusion",
    "observationPack",
    "evidencePreservingReducer",
    "evidencePreservingReducerProvider",
    "evidencePreservingReducerModel",
    "onlineContextCompact",
    "cacheWriteReadRatio",
}


class SharedPolicyTests(unittest.TestCase):
    def test_schema_version_1_keys_match_sol_pi(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        props = schema["properties"]
        self.assertEqual(set(props), OPERATOR_KEYS)
        self.assertEqual(props["version"]["const"], 1)
        for key in (
            "actionFusion",
            "observationPack",
            "evidencePreservingReducer",
            "onlineContextCompact",
        ):
            self.assertEqual(props[key]["type"], "boolean")
        self.assertFalse(schema.get("additionalProperties", True))
        constants = schema["$defs"]["observationPackConstants"]["properties"]
        self.assertEqual(constants["FULL_SENDS"]["const"], 2)
        self.assertEqual(constants["thresholdBytes"]["const"], 10240)

    def test_observation_pack_constants_match_schema_defs(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        constants = schema["$defs"]["observationPackConstants"]["properties"]
        self.assertEqual(FULL_SENDS, constants["FULL_SENDS"]["const"])
        self.assertEqual(THRESHOLD_BYTES, constants["thresholdBytes"]["const"])
        self.assertEqual(FULL_SENDS, 2)
        self.assertEqual(THRESHOLD_BYTES, 10240)

    def test_canonical_instance_has_only_operator_keys(self) -> None:
        instance = json.loads(INSTANCE_PATH.read_text(encoding="utf-8"))
        self.assertTrue(set(instance).issubset(OPERATOR_KEYS))
        self.assertNotIn("FULL_SENDS", instance)
        self.assertNotIn("thresholdBytes", instance)
        self.assertEqual(instance["version"], 1)
        self.assertTrue(instance["actionFusion"])
        self.assertTrue(instance["observationPack"])
        self.assertFalse(instance["evidencePreservingReducer"])
        self.assertFalse(instance["onlineContextCompact"])

    def test_hermes_loader_accepts_canonical_policy_file(self) -> None:
        payload = INSTANCE_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "hermes-home"
            home.mkdir()
            (home / "sol-pi.json").write_text(payload, encoding="utf-8")
            cfg = load_sol_pi_config(cwd=None, hermes_home=home)
        self.assertEqual(cfg.version, 1)
        self.assertTrue(cfg.action_fusion)
        self.assertTrue(cfg.observation_pack)
        self.assertFalse(cfg.evidence_preserving_reducer)
        self.assertFalse(cfg.online_context_compact)

    def test_hermes_loader_accepts_canonical_policy_raw(self) -> None:
        raw = json.loads(INSTANCE_PATH.read_text(encoding="utf-8"))
        cfg = load_sol_pi_config(raw=raw)
        self.assertTrue(cfg.action_fusion)
        self.assertTrue(cfg.observation_pack)


if __name__ == "__main__":
    unittest.main()
