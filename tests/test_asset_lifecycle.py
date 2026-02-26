"""Integration tests for the full asset registration lifecycle.

Exercises the end-to-end flow: register → verify routing → unregister → verify cleanup.
"""

import json
from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import list_assets, register_asset, unregister_asset


class TestAssetLifecycle:
    """End-to-end lifecycle: register → route → unregister → verify cleanup."""

    def test_full_lifecycle(self, tmp_path):
        """Register a model, verify routing, unregister, verify cleanup."""
        # --- Setup ---
        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {
                "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            },
            "roles": {
                "coding": ["m1"],
            },
        }
        fabric = ComputeFabric(config)

        gguf_file = tmp_path / "m2.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        mock_storage = MagicMock()
        mock_storage.list_models.return_value = []
        mock_storage.remove.return_value = True

        # --- Register ---
        with patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage), \
             patch("aurarouter.tuning.extract_gguf_metadata", return_value={"context_length": 8192, "architecture": "llama"}):
            result = register_asset(
                fabric, config,
                model_id="m2",
                file_path=str(gguf_file),
                tags="coding",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "coding" in parsed["roles_joined"]

        # --- Verify routing config ---
        assert config.get_role_chain("coding") == ["m1", "m2"]

        # --- Verify live fabric has updated config ---
        assert fabric._config.get_role_chain("coding") == ["m1", "m2"]
        assert "m2" in fabric._config.get_all_model_ids()

        # --- Verify metadata-derived parameters ---
        m2_cfg = config.get_model_config("m2")
        assert m2_cfg["parameters"]["n_ctx"] == 8192

        # --- Verify list_assets reflects the new model ---
        mock_storage.list_models.return_value = [
            {"repo": "local", "filename": "m2.gguf", "path": str(gguf_file), "size_bytes": 1024},
        ]
        with patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            assets_result = list_assets()
        assets = json.loads(assets_result)
        assert len(assets) == 1
        assert assets[0]["filename"] == "m2.gguf"

        # --- Unregister ---
        with patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            unreg_result = unregister_asset(
                fabric, config,
                model_id="m2",
            )

        unreg_parsed = json.loads(unreg_result)
        assert unreg_parsed["success"] is True
        assert "coding" in unreg_parsed["roles_left"]

        # --- Verify cleanup ---
        assert config.get_role_chain("coding") == ["m1"]
        assert fabric._config.get_role_chain("coding") == ["m1"]
        assert "m2" not in config.get_all_model_ids()

        # --- Verify list_assets post-unregister ---
        mock_storage.list_models.return_value = []
        with patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            assets_after = list_assets()
        assert json.loads(assets_after) == []


class TestConcurrentRegistration:
    """Verify multiple registrations to the same role chain don't corrupt it."""

    def test_two_models_same_tag_append_correctly(self, tmp_path):
        """Register two models with same tag — both append, no duplicates."""
        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {
                "m1": {"provider": "ollama", "model_name": "test"},
            },
            "roles": {
                "coding": ["m1"],
            },
        }
        fabric = ComputeFabric(config)

        gguf_a = tmp_path / "model_a.gguf"
        gguf_a.write_bytes(b"\x00" * 1024)
        gguf_b = tmp_path / "model_b.gguf"
        gguf_b.write_bytes(b"\x00" * 1024)

        mock_storage = MagicMock()

        with patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):

            # Register model_a
            result_a = register_asset(
                fabric, config,
                model_id="model_a",
                file_path=str(gguf_a),
                tags="coding",
            )
            parsed_a = json.loads(result_a)
            assert parsed_a["success"] is True
            assert "coding" in parsed_a["roles_joined"]

            # Register model_b
            result_b = register_asset(
                fabric, config,
                model_id="model_b",
                file_path=str(gguf_b),
                tags="coding",
            )
            parsed_b = json.loads(result_b)
            assert parsed_b["success"] is True
            assert "coding" in parsed_b["roles_joined"]

        # Verify final chain: original + both new models, no duplicates
        chain = config.get_role_chain("coding")
        assert chain == ["m1", "model_a", "model_b"]
        assert len(chain) == len(set(chain))  # no duplicates
