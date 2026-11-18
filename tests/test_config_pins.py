import json
import tempfile
import unittest
from pathlib import Path

from sol_pi_hermes.config import (
    DEFAULT_CACHE_WRITE_READ_RATIO,
    DEFAULT_REDUCER_MODEL,
    DEFAULT_REDUCER_PROVIDER,
    SolPiConfig,
    find_config_path,
    load_sol_pi_config,
)


def _write(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sol-pi.json"
    path.write_text(body, encoding="utf-8")
    return path


class ConfigPathTests(unittest.TestCase):
    def test_project_file_wins_over_global_and_missing_means_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home, cwd = Path(tmp) / "home", Path(tmp) / "proj"
            home.mkdir()
            self.assertIsNone(find_config_path(cwd, home))
            self.assertEqual(load_sol_pi_config(cwd, home), SolPiConfig())
            glob = _write(home, json.dumps({"version": 1, "actionFusion": True}))
            self.assertEqual(find_config_path(cwd, home), glob)
            self.assertEqual(find_config_path(None, home), glob)
            proj = _write(cwd / ".hermes", json.dumps({"version": 1, "observationPack": True}))
            self.assertEqual(find_config_path(cwd, home), proj)
            cfg = load_sol_pi_config(cwd, home)
            self.assertTrue(cfg.observation_pack)
            self.assertFalse(cfg.action_fusion)

    def test_project_config_that_is_a_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home, cwd = Path(tmp) / "home", Path(tmp) / "proj"
            glob = _write(home, json.dumps({"version": 1, "actionFusion": True}))
            (cwd / ".hermes" / "sol-pi.json").mkdir(parents=True)
            self.assertEqual(find_config_path(cwd, home), glob)
            self.assertTrue(load_sol_pi_config(cwd, home).action_fusion)

    def test_unreadable_json_is_a_named_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            _write(home, "{not json")
            with self.assertRaisesRegex(ValueError, "Unable to read SoL-Pi config"):
                load_sol_pi_config(None, home)


class ConfigValidationTests(unittest.TestCase):
    def test_non_object_rejected(self) -> None:
        for raw in ([], "x", 3):
            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                load_sol_pi_config(raw=raw)  # type: ignore[arg-type]

    def test_version_must_be_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "version must be 1"):
            load_sol_pi_config(raw={"version": 2})

    def test_each_feature_flag_is_read_and_must_be_boolean(self) -> None:
        fields = {
            "actionFusion": "action_fusion",
            "observationPack": "observation_pack",
            "evidencePreservingReducer": "evidence_preserving_reducer",
            "onlineContextCompact": "online_context_compact",
        }
        for key, attr in fields.items():
            cfg = load_sol_pi_config(raw={"version": 1, key: True})
            self.assertEqual(
                {a for a in fields.values() if getattr(cfg, a)}, {attr}, key
            )
            for bad in (1, "true", None):
                with self.assertRaisesRegex(ValueError, f"{key} must be boolean"):
                    load_sol_pi_config(raw={"version": 1, key: bad})

    def test_reducer_strings_default_override_and_reject_blank(self) -> None:
        cfg = load_sol_pi_config(raw={"version": 1})
        self.assertEqual(cfg.evidence_preserving_reducer_model, DEFAULT_REDUCER_MODEL)
        self.assertEqual(cfg.evidence_preserving_reducer_provider, DEFAULT_REDUCER_PROVIDER)
        cfg = load_sol_pi_config(raw={
            "version": 1,
            "evidencePreservingReducerModel": "m",
            "evidencePreservingReducerProvider": "p",
        })
        self.assertEqual((cfg.evidence_preserving_reducer_model, cfg.evidence_preserving_reducer_provider), ("m", "p"))
        for key in ("evidencePreservingReducerModel", "evidencePreservingReducerProvider"):
            for bad in ("", "  ", 5, None):
                with self.assertRaisesRegex(ValueError, f"{key} must be a non-empty string"):
                    load_sol_pi_config(raw={"version": 1, key: bad})

    def test_cache_ratio_default_valid_and_invalid(self) -> None:
        self.assertEqual(load_sol_pi_config(raw={"version": 1}).cache_write_read_ratio, DEFAULT_CACHE_WRITE_READ_RATIO)
        for good, want in ((0, 0.0), (3, 3.0), (1.5, 1.5)):
            cfg = load_sol_pi_config(raw={"version": 1, "cacheWriteReadRatio": good})
            self.assertEqual(cfg.cache_write_read_ratio, want)
            self.assertIsInstance(cfg.cache_write_read_ratio, float)
        for bad in (-1, -0.5, True, False, "2", None):
            with self.assertRaisesRegex(ValueError, "cacheWriteReadRatio"):
                load_sol_pi_config(raw={"version": 1, "cacheWriteReadRatio": bad})


if __name__ == "__main__":
    unittest.main()
