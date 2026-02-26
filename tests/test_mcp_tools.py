"""Tests for MCP tool implementations (mcp_tools.py)."""

import json
from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import (
    compare_models,
    generate_code,
    list_assets,
    local_inference,
    register_asset,
    route_task,
    unregister_asset,
)
from aurarouter.savings.models import GenerateResult


def _make_fabric(models=None, roles=None) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": models or {
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            "m2": {"provider": "google", "model_name": "gemini-flash", "api_key": "k"},
        },
        "roles": roles or {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1", "m2"],
        },
    }
    return ComputeFabric(cfg)


# ------------------------------------------------------------------
# route_task
# ------------------------------------------------------------------

class TestRouteTask:
    def test_simple_intent(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE", "complexity": 3}),
            "result text",
        ]):
            result = route_task(fabric, None, task="hello world")
            assert result == "result text"

    def test_complex_intent(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "COMPLEX_REASONING", "complexity": 8}),
            json.dumps(["step 1", "step 2"]),
            "step 1 output",
            "step 2 output",
        ]):
            result = route_task(fabric, None, task="complex task")
            assert "Step 1" in result
            assert "Step 2" in result

    def test_all_models_fail(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE"}),
            None,
        ]):
            result = route_task(fabric, None, task="test")
            assert "Error" in result


# ------------------------------------------------------------------
# local_inference
# ------------------------------------------------------------------

class TestLocalInference:
    def test_filters_to_local_only(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute") as mock:
            mock.return_value = "local result"
            result = local_inference(fabric, prompt="test")
            assert result == "local result"
            # Verify chain_override was passed with only local models
            call_kwargs = mock.call_args
            override = call_kwargs.kwargs.get("chain_override")
            assert override is not None
            assert "m1" in override      # ollama (local)
            assert "m2" not in override  # google (cloud)

    def test_error_when_no_local_models(self):
        fabric = _make_fabric(
            models={"m2": {"provider": "google", "model_name": "g", "api_key": "k"}},
            roles={"coding": ["m2"]},
        )
        result = local_inference(fabric, prompt="test")
        assert "Error" in result
        assert "local" in result.lower()

    def test_includes_context(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute") as mock:
            mock.return_value = "ok"
            local_inference(fabric, prompt="test", context="extra context")
            prompt_sent = mock.call_args.args[1]
            assert "extra context" in prompt_sent


# ------------------------------------------------------------------
# generate_code
# ------------------------------------------------------------------

class TestGenerateCode:
    def test_simple_code(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "SIMPLE_CODE"}),
            "def add(a, b): return a + b",
        ]):
            result = generate_code(
                fabric, None,
                task_description="write an add function",
                language="python",
            )
            assert "def add" in result

    def test_complex_code(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", side_effect=[
            json.dumps({"intent": "COMPLEX_REASONING"}),
            json.dumps(["create module", "add tests"]),
            "# module code",
            "# test code",
        ]):
            result = generate_code(
                fabric, None,
                task_description="build a module with tests",
            )
            assert "Step 1" in result
            assert "Step 2" in result


# ------------------------------------------------------------------
# compare_models
# ------------------------------------------------------------------

class TestCompareModels:
    def test_returns_all_results(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute_all", return_value=[
            {"model_id": "m1", "provider": "ollama", "success": True,
             "text": "result1", "elapsed_s": 1.0, "input_tokens": 10, "output_tokens": 20},
            {"model_id": "m2", "provider": "google", "success": True,
             "text": "result2", "elapsed_s": 2.0, "input_tokens": 10, "output_tokens": 30},
        ]):
            result = compare_models(fabric, prompt="test")
            assert "m1" in result
            assert "m2" in result
            assert "result1" in result
            assert "result2" in result
            assert "SUCCESS" in result

    def test_empty_results(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute_all", return_value=[]):
            result = compare_models(fabric, prompt="test")
            assert "Error" in result

    def test_passes_model_ids(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute_all") as mock:
            mock.return_value = []
            compare_models(fabric, prompt="test", models="m1, m2")
            call_kwargs = mock.call_args
            assert call_kwargs.kwargs.get("model_ids") == ["m1", "m2"]


# ------------------------------------------------------------------
# list_assets
# ------------------------------------------------------------------

class TestListAssets:
    """Tests for the list_assets MCP tool."""

    def test_list_assets_empty_registry(self, tmp_path):
        """Returns empty array when no models are registered."""
        mock_storage = MagicMock(list_models=MagicMock(return_value=[]))
        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            return_value=mock_storage,
        ):
            result = list_assets()
        parsed = json.loads(result)
        assert parsed == []

    def test_list_assets_with_models(self, tmp_path):
        """Returns array of asset entries when models exist."""
        entries = [
            {
                "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                "filename": "qwen-7b.gguf",
                "path": str(tmp_path / "qwen-7b.gguf"),
                "size_bytes": 4_000_000_000,
                "downloaded_at": "2026-02-20T12:00:00+00:00",
            },
            {
                "repo": "TheBloke/Llama-2-7B-GGUF",
                "filename": "llama-7b.gguf",
                "path": str(tmp_path / "llama-7b.gguf"),
                "size_bytes": 3_500_000_000,
                "downloaded_at": "2026-02-21T12:00:00+00:00",
            },
        ]
        mock_storage = MagicMock(list_models=MagicMock(return_value=entries))
        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            return_value=mock_storage,
        ):
            result = list_assets()
        parsed = json.loads(result)
        assert len(parsed) == 2
        for entry in parsed:
            assert "repo" in entry
            assert "filename" in entry
            assert "path" in entry
            assert "size_bytes" in entry
            assert "downloaded_at" in entry

    def test_list_assets_includes_metadata(self, tmp_path):
        """Includes gguf_metadata when present in registry."""
        entries = [
            {
                "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
                "filename": "qwen-7b.gguf",
                "path": str(tmp_path / "qwen-7b.gguf"),
                "size_bytes": 4_000_000_000,
                "downloaded_at": "2026-02-20T12:00:00+00:00",
                "gguf_metadata": {"context_length": 32768, "quantization": "Q4_K_M"},
            },
        ]
        mock_storage = MagicMock(list_models=MagicMock(return_value=entries))
        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            return_value=mock_storage,
        ):
            result = list_assets()
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert "gguf_metadata" in parsed[0]
        assert parsed[0]["gguf_metadata"]["context_length"] == 32768

    def test_list_assets_handles_error(self):
        """Returns error JSON when storage access fails."""
        with patch(
            "aurarouter.models.file_storage.FileModelStorage",
            side_effect=OSError("Permission denied"),
        ):
            result = list_assets()
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Permission denied" in parsed["error"]


# ------------------------------------------------------------------
# register_asset
# ------------------------------------------------------------------

class TestRegisterAsset:
    """Tests for the register_asset MCP tool."""

    def test_register_new_model_success(self, tmp_path):
        """Successfully registers a new model and updates config."""
        gguf_file = tmp_path / "test-model.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)
        mock_storage = MagicMock()

        with patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage), \
             patch.object(config, "save"), \
             patch.object(fabric, "update_config") as mock_update, \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("not real gguf")):
            result = register_asset(
                fabric, config,
                model_id="test-model",
                file_path=str(gguf_file),
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["model_id"] == "test-model"
        assert parsed["path"] == str(gguf_file)

        # Verify model was added to config
        assert "test-model" in config.get_all_model_ids()
        model_cfg = config.get_model_config("test-model")
        assert model_cfg["provider"] == "llamacpp"

        # Verify live fabric was updated
        mock_update.assert_called_once_with(config)

        # Verify FileModelStorage was called
        mock_storage.register.assert_called_once()

    def test_register_duplicate_model_id(self, tmp_path):
        """Returns error when model_id already exists."""
        gguf_file = tmp_path / "test-model.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.set_model("existing-model", {"provider": "llamacpp"})
        fabric = ComputeFabric(config)

        result = register_asset(
            fabric, config,
            model_id="existing-model",
            file_path=str(gguf_file),
        )

        parsed = json.loads(result)
        assert "error" in parsed
        assert "already exists" in parsed["error"]

    def test_register_nonexistent_file(self):
        """Returns error when file_path does not exist."""
        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)

        result = register_asset(
            fabric, config,
            model_id="ghost-model",
            file_path="/nonexistent/path/model.gguf",
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_register_with_tags(self, tmp_path):
        """Tags are correctly parsed and stored in config."""
        gguf_file = tmp_path / "tagged-model.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)
        mock_storage = MagicMock()

        with patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage), \
             patch.object(config, "save"), \
             patch.object(fabric, "update_config"), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("not real")):
            result = register_asset(
                fabric, config,
                model_id="tagged-model",
                file_path=str(gguf_file),
                tags="coding,local,fine-tuned",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True

        model_cfg = config.get_model_config("tagged-model")
        assert model_cfg["tags"] == ["coding", "local", "fine-tuned"]

    def test_register_non_gguf_file(self, tmp_path):
        """Returns error when file is not a .gguf file."""
        txt_file = tmp_path / "model.txt"
        txt_file.write_text("not a model")

        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)

        result = register_asset(
            fabric, config,
            model_id="bad-model",
            file_path=str(txt_file),
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert ".gguf" in parsed["error"].lower() or "invalid" in parsed["error"].lower()

    def test_register_updates_live_fabric(self, tmp_path):
        """Verifies fabric.update_config is called on successful registration."""
        gguf_file = tmp_path / "new-model.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config") as mock_update, \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("not real")):
            result = register_asset(
                fabric, config,
                model_id="new-model",
                file_path=str(gguf_file),
            )
        parsed = json.loads(result)
        assert parsed["success"] is True
        mock_update.assert_called_once_with(config)
        assert "new-model" in config.get_all_model_ids()


# ------------------------------------------------------------------
# register_asset — tag-to-role auto-integration
# ------------------------------------------------------------------

class TestRegisterAssetRoleIntegration:
    """Tests for tag-to-role chain auto-integration in register_asset."""

    def test_tag_matches_existing_role(self, tmp_path):
        """Model with tag matching an existing role joins that role chain."""
        gguf_file = tmp_path / "coder.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {"m1": {"provider": "ollama"}},
            "roles": {"coding": ["m1"]},
        }
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):
            result = register_asset(
                fabric, config,
                model_id="coder-model",
                file_path=str(gguf_file),
                tags="coding",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "coding" in parsed["roles_joined"]
        assert config.get_role_chain("coding") == ["m1", "coder-model"]

    def test_tag_matches_semantic_synonym(self, tmp_path):
        """Model with tag matching a semantic verb synonym joins the role."""
        gguf_file = tmp_path / "prog.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {"m1": {"provider": "ollama"}},
            "roles": {"coding": ["m1"]},
            "semantic_verbs": {"coding": {"synonyms": ["programming", "development"]}},
        }
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):
            result = register_asset(
                fabric, config,
                model_id="prog-model",
                file_path=str(gguf_file),
                tags="programming",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "coding" in parsed["roles_joined"]
        assert config.get_role_chain("coding") == ["m1", "prog-model"]

    def test_multiple_tags_multiple_roles(self, tmp_path):
        """Model with multiple matching tags joins multiple role chains."""
        gguf_file = tmp_path / "multi.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {"m1": {"provider": "ollama"}},
            "roles": {"coding": ["m1"], "reasoning": ["m1"]},
        }
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):
            result = register_asset(
                fabric, config,
                model_id="multi-model",
                file_path=str(gguf_file),
                tags="coding,reasoning",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert sorted(parsed["roles_joined"]) == ["coding", "reasoning"]
        assert config.get_role_chain("coding") == ["m1", "multi-model"]
        assert config.get_role_chain("reasoning") == ["m1", "multi-model"]

    def test_tag_no_matching_role(self, tmp_path):
        """Model with non-matching tags still registers successfully."""
        gguf_file = tmp_path / "exp.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {"m1": {"provider": "ollama"}},
            "roles": {"coding": ["m1"]},
        }
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):
            result = register_asset(
                fabric, config,
                model_id="exp-model",
                file_path=str(gguf_file),
                tags="experimental",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["roles_joined"] == []
        assert config.get_role_chain("coding") == ["m1"]  # unchanged

    def test_duplicate_prevention_in_role_chain(self, tmp_path):
        """Model already in a role chain is not added again."""
        gguf_file = tmp_path / "dup.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {"m1": {"provider": "ollama"}, "dup-model": {"provider": "llamacpp"}},
            "roles": {"coding": ["m1", "dup-model"]},
        }
        fabric = ComputeFabric(config)

        # The model_id already exists check will catch this first
        result = register_asset(
            fabric, config,
            model_id="dup-model",
            file_path=str(gguf_file),
            tags="coding",
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "already exists" in parsed["error"]

    def test_response_includes_roles_joined(self, tmp_path):
        """The JSON response includes the roles_joined array."""
        gguf_file = tmp_path / "rj.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {"coding": []}}
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):
            result = register_asset(
                fabric, config,
                model_id="rj-model",
                file_path=str(gguf_file),
                tags="coding",
            )

        parsed = json.loads(result)
        assert "roles_joined" in parsed
        assert isinstance(parsed["roles_joined"], list)
        assert "coding" in parsed["roles_joined"]


# ------------------------------------------------------------------
# register_asset — GGUF metadata extraction
# ------------------------------------------------------------------

class TestRegisterAssetMetadata:
    """Tests for GGUF metadata extraction during registration."""

    def test_metadata_extracted_on_registration(self, tmp_path):
        """GGUF metadata is extracted and used for model config parameters."""
        gguf_file = tmp_path / "meta.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)
        mock_storage = MagicMock()
        fake_metadata = {"context_length": 32768, "architecture": "qwen2"}

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage), \
             patch("aurarouter.tuning.extract_gguf_metadata", return_value=fake_metadata):
            result = register_asset(
                fabric, config,
                model_id="meta-model",
                file_path=str(gguf_file),
            )

        parsed = json.loads(result)
        assert parsed["success"] is True

        # Verify metadata-derived parameters in config
        model_cfg = config.get_model_config("meta-model")
        assert model_cfg["parameters"]["n_ctx"] == 32768

        # Verify storage.register was called with metadata
        mock_storage.register.assert_called_once()
        call_kwargs = mock_storage.register.call_args
        assert call_kwargs.kwargs.get("metadata") == fake_metadata or \
               call_kwargs[1].get("metadata") == fake_metadata or \
               (len(call_kwargs[0]) >= 4 and call_kwargs[0][3] == fake_metadata)

    def test_metadata_extraction_failure_nonfatal(self, tmp_path):
        """Registration succeeds even when metadata extraction fails."""
        gguf_file = tmp_path / "bad-meta.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)
        mock_storage = MagicMock()

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("corrupt")):
            result = register_asset(
                fabric, config,
                model_id="bad-meta-model",
                file_path=str(gguf_file),
            )

        parsed = json.loads(result)
        assert parsed["success"] is True

        # No parameters.n_ctx should be set
        model_cfg = config.get_model_config("bad-meta-model")
        assert "parameters" not in model_cfg


# ------------------------------------------------------------------
# register_asset — cost & hosting tier fields (TG4)
# ------------------------------------------------------------------

class TestRegisterAssetCostFields:
    """Tests for cost and hosting tier fields in register_asset."""

    def test_register_asset_with_cost_fields(self, tmp_path):
        """register_asset with cost fields includes them in config."""
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)

        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):
            result_json = register_asset(
                fabric, config,
                model_id="test-model",
                file_path=str(gguf),
                cost_per_1m_input=0.50,
                cost_per_1m_output=2.00,
                hosting_tier="on-prem",
            )
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["cost_per_1m_input"] == 0.50
        assert result["cost_per_1m_output"] == 2.00
        assert result["hosting_tier"] == "on-prem"

        model_cfg = config.get_model_config("test-model")
        assert model_cfg["cost_per_1m_input"] == 0.50
        assert model_cfg["cost_per_1m_output"] == 2.00
        assert model_cfg["hosting_tier"] == "on-prem"

    def test_register_asset_without_cost_fields(self, tmp_path):
        """register_asset without cost fields omits them from config."""
        gguf = tmp_path / "test.gguf"
        gguf.write_bytes(b"\x00" * 100)

        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()), \
             patch("aurarouter.tuning.extract_gguf_metadata", side_effect=ValueError("skip")):
            result_json = register_asset(
                fabric, config,
                model_id="test-model",
                file_path=str(gguf),
            )
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["cost_per_1m_input"] is None
        assert result["cost_per_1m_output"] is None
        assert result["hosting_tier"] is None

        model_cfg = config.get_model_config("test-model")
        assert "cost_per_1m_input" not in model_cfg
        assert "cost_per_1m_output" not in model_cfg
        assert "hosting_tier" not in model_cfg


# ------------------------------------------------------------------
# unregister_asset
# ------------------------------------------------------------------

class TestUnregisterAsset:
    """Tests for the unregister_asset MCP tool."""

    def test_unregister_success(self, tmp_path):
        """Successfully unregisters a model from config and role chains."""
        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {"m1": {"provider": "ollama"}, "m2": {"provider": "llamacpp", "model_path": str(tmp_path / "m2.gguf")}},
            "roles": {"coding": ["m1", "m2"]},
        }
        fabric = ComputeFabric(config)
        mock_storage = MagicMock()
        mock_storage.remove.return_value = True

        with patch.object(fabric, "update_config") as mock_update, \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            result = unregister_asset(
                fabric, config,
                model_id="m2",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["model_id"] == "m2"
        assert "coding" in parsed["roles_left"]
        assert parsed["file_deleted"] is False

        # Verify model removed from config
        assert "m2" not in config.get_all_model_ids()
        # Verify model removed from role chain
        assert config.get_role_chain("coding") == ["m1"]
        # Verify fabric was updated
        mock_update.assert_called_once_with(config)

    def test_unregister_nonexistent_model(self):
        """Returns error when model_id does not exist."""
        config = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(config)

        result = unregister_asset(
            fabric, config,
            model_id="ghost",
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_unregister_keep_roles(self, tmp_path):
        """With remove_from_roles=False, role chains are untouched."""
        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {"m1": {"provider": "ollama"}, "m2": {"provider": "llamacpp", "model_path": str(tmp_path / "m2.gguf")}},
            "roles": {"coding": ["m1", "m2"]},
        }
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()):
            result = unregister_asset(
                fabric, config,
                model_id="m2",
                remove_from_roles=False,
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["roles_left"] == []
        # Model removed from config but chain still has it
        assert "m2" not in config.get_all_model_ids()
        assert "m2" in config.get_role_chain("coding")

    def test_unregister_delete_file(self, tmp_path):
        """With delete_file=True, storage.remove is called with delete_file=True."""
        gguf_file = tmp_path / "deleteme.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        config = ConfigLoader(allow_missing=True)
        config.set_model("del-model", {"provider": "llamacpp", "model_path": str(gguf_file)})
        fabric = ComputeFabric(config)
        mock_storage = MagicMock()
        mock_storage.remove.return_value = True

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            result = unregister_asset(
                fabric, config,
                model_id="del-model",
                delete_file=True,
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["file_deleted"] is True
        mock_storage.remove.assert_called_once_with("deleteme.gguf", delete_file=True)

    def test_unregister_removes_from_multiple_roles(self, tmp_path):
        """Model in multiple role chains is removed from all of them."""
        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {
                "m1": {"provider": "ollama"},
                "m2": {"provider": "llamacpp", "model_path": str(tmp_path / "m2.gguf")},
            },
            "roles": {"coding": ["m1", "m2"], "reasoning": ["m2", "m1"]},
        }
        fabric = ComputeFabric(config)

        with patch.object(fabric, "update_config"), \
             patch.object(config, "save"), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=MagicMock()):
            result = unregister_asset(
                fabric, config,
                model_id="m2",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert sorted(parsed["roles_left"]) == ["coding", "reasoning"]
        assert config.get_role_chain("coding") == ["m1"]
        assert config.get_role_chain("reasoning") == ["m1"]
