"""V1 shared sol-pi.json operator policy and Hermes loader contract.

The schema comes from sol-pi-hermes PR #1, with the existing non-whitespace
model route requirement made explicit. No jsonschema runtime dependency.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.config import CONFIG_KEYS, load_sol_pi_config
from sol_pi_hermes.observation_pack import FULL_SENDS, THRESHOLD_BYTES

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schema" / "sol-pi.policy.v1.schema.json"
INSTANCE = ROOT / "schema" / "sol-pi.policy.v1.json"


class SharedPolicyLoaderTests(unittest.TestCase):
    def test_schema_and_runtime_operator_keys_stay_in_sync(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(set(schema["properties"]), CONFIG_KEYS)
        self.assertEqual(schema["required"], ["version"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["version"]["const"], 1)
        for key in ("actionFusion", "observationPack",
                    "evidencePreservingReducer", "onlineContextCompact"):
            self.assertEqual(schema["properties"][key]["type"], "boolean")
        for key in ("evidencePreservingReducerProvider",
                    "evidencePreservingReducerModel"):
            self.assertEqual(schema["properties"][key]["type"], "string")
            self.assertEqual(schema["properties"][key]["minLength"], 1)
            self.assertEqual(schema["properties"][key]["pattern"], r"\S")
        self.assertEqual(schema["properties"]["cacheWriteReadRatio"]["type"], "number")
        self.assertEqual(schema["properties"]["cacheWriteReadRatio"]["minimum"], 0)

    def test_observation_pack_constants_are_not_operator_keys(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        constants = schema["$defs"]["observationPackConstants"]["properties"]
        self.assertEqual(constants["FULL_SENDS"]["const"], FULL_SENDS)
        self.assertEqual(constants["thresholdBytes"]["const"], THRESHOLD_BYTES)
        self.assertEqual((FULL_SENDS, THRESHOLD_BYTES), (2, 10240))
        self.assertFalse({"FULL_SENDS", "thresholdBytes"} & CONFIG_KEYS)
        for key in ("FULL_SENDS", "thresholdBytes"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unknown"):
                load_sol_pi_config(raw={"version": 1, key: 2})

    def test_shared_example_loads_as_file_and_raw(self) -> None:
        policy = json.loads(INSTANCE.read_text(encoding="utf-8"))
        self.assertEqual(set(policy) - CONFIG_KEYS, set())
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home / "sol-pi.json").write_text(
                INSTANCE.read_text(encoding="utf-8"), encoding="utf-8"
            )
            file_cfg = load_sol_pi_config(hermes_home=home)
        raw_cfg = load_sol_pi_config(raw=policy)
        self.assertEqual(file_cfg, raw_cfg)
        self.assertTrue(raw_cfg.action_fusion)
        self.assertTrue(raw_cfg.observation_pack)
        self.assertFalse(raw_cfg.evidence_preserving_reducer)
        self.assertFalse(raw_cfg.online_context_compact)

    def test_no_file_and_legacy_empty_raw_mean_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = load_sol_pi_config(hermes_home=Path(temp))
        self.assertEqual(missing, load_sol_pi_config(raw={}))
        self.assertEqual(missing.version, 1)
        self.assertFalse(missing.action_fusion)
        self.assertFalse(missing.observation_pack)
        self.assertFalse(missing.evidence_preserving_reducer)
        self.assertFalse(missing.online_context_compact)
        self.assertEqual(missing.cache_write_read_ratio, 12.5)

    def test_present_file_must_declare_version_one(self) -> None:
        for payload in ('{}', '{"actionFusion": true}'):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                home = Path(temp)
                (home / "sol-pi.json").write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "version is required"):
                    load_sol_pi_config(hermes_home=home)

        with self.assertRaisesRegex(ValueError, "version is required"):
            load_sol_pi_config(raw={"actionFusion": True})

    def test_version_true_is_not_integer_one(self) -> None:
        for value in (True, False, "1", 2, None, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "version must be 1"):
                load_sol_pi_config(raw={"version": value})
        # JSON Schema treats equivalent numeric representations of const 1 equally.
        self.assertEqual(load_sol_pi_config(raw={"version": 1.0}).version, 1)

    def test_wrong_feature_types_and_unknown_keys_rejected(self) -> None:
        for key in ("actionFusion", "observationPack",
                    "evidencePreservingReducer", "onlineContextCompact"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_sol_pi_config(raw={"version": 1, key: 1})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_sol_pi_config(raw={"version": 1, "other": False})

    def test_ratio_must_be_finite_nonnegative_numeric(self) -> None:
        for value in (float("inf"), float("-inf"), float("nan"),
                      -0.01, True, "12.5", 10 ** 400):
            with self.subTest(value=str(value)[:40]), self.assertRaisesRegex(ValueError, "finite non-negative"):
                load_sol_pi_config(raw={"version": 1, "cacheWriteReadRatio": value})
        cfg = load_sol_pi_config(raw={"version": 1, "cacheWriteReadRatio": 0})
        self.assertEqual(cfg.cache_write_read_ratio, 0.0)
        self.assertEqual(load_sol_pi_config(raw={"version": 1}).cache_write_read_ratio, 12.5)

    def test_overflowing_json_ratio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home / "sol-pi.json").write_text(
                '{"version": 1, "cacheWriteReadRatio": 1e400}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "finite non-negative"):
                load_sol_pi_config(hermes_home=home)

    def test_route_values_cannot_be_blank(self) -> None:
        for key in ("evidencePreservingReducerProvider", "evidencePreservingReducerModel"):
            for value in ("", "  ", 0):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_sol_pi_config(raw={"version": 1, key: value})

    def test_valid_operator_values_are_carried_through(self) -> None:
        cfg = load_sol_pi_config(raw={
            "version": 1, "actionFusion": False, "observationPack": True,
            "evidencePreservingReducer": True, "onlineContextCompact": False,
            "evidencePreservingReducerProvider": "prov-x",
            "evidencePreservingReducerModel": "model-y", "cacheWriteReadRatio": 3})
        self.assertFalse(cfg.action_fusion)
        self.assertTrue(cfg.observation_pack)
        self.assertTrue(cfg.evidence_preserving_reducer)
        self.assertFalse(cfg.online_context_compact)
        self.assertEqual((cfg.evidence_preserving_reducer_provider,
                          cfg.evidence_preserving_reducer_model), ("prov-x", "model-y"))
        self.assertEqual(cfg.cache_write_read_ratio, 3.0)
        self.assertIsInstance(cfg.cache_write_read_ratio, float)


if __name__ == "__main__":
    unittest.main()
