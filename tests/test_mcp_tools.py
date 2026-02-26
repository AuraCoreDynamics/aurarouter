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
        # Create a dummy .gguf file
        gguf_file = tmp_path / "test-model.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        mock_config = MagicMock()
        mock_config.get_model_config.return_value = {}  # model_id does not exist
        mock_storage = MagicMock()

        with patch("aurarouter.config.ConfigLoader", return_value=mock_config), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            result = register_asset(
                model_id="test-model",
                file_path=str(gguf_file),
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["model_id"] == "test-model"
        assert parsed["path"] == str(gguf_file)

        # Verify config was updated
        mock_config.set_model.assert_called_once()
        call_args = mock_config.set_model.call_args
        assert call_args[0][0] == "test-model"
        assert call_args[0][1]["provider"] == "llamacpp"
        mock_config.save.assert_called_once()

        # Verify FileModelStorage was called
        mock_storage.register.assert_called_once()

    def test_register_duplicate_model_id(self, tmp_path):
        """Returns error when model_id already exists."""
        gguf_file = tmp_path / "test-model.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)

        mock_config = MagicMock()
        mock_config.get_model_config.return_value = {"provider": "llamacpp"}  # already exists

        with patch("aurarouter.config.ConfigLoader", return_value=mock_config):
            result = register_asset(
                model_id="existing-model",
                file_path=str(gguf_file),
            )

        parsed = json.loads(result)
        assert "error" in parsed
        assert "already exists" in parsed["error"]
        mock_config.set_model.assert_not_called()

    def test_register_nonexistent_file(self):
        """Returns error when file_path does not exist."""
        result = register_asset(
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

        mock_config = MagicMock()
        mock_config.get_model_config.return_value = {}
        mock_storage = MagicMock()

        with patch("aurarouter.config.ConfigLoader", return_value=mock_config), \
             patch("aurarouter.models.file_storage.FileModelStorage", return_value=mock_storage):
            result = register_asset(
                model_id="tagged-model",
                file_path=str(gguf_file),
                tags="coding,local,fine-tuned",
            )

        parsed = json.loads(result)
        assert parsed["success"] is True

        call_args = mock_config.set_model.call_args
        model_cfg = call_args[0][1]
        assert model_cfg["tags"] == ["coding", "local", "fine-tuned"]

    def test_register_non_gguf_file(self, tmp_path):
        """Returns error when file is not a .gguf file."""
        txt_file = tmp_path / "model.txt"
        txt_file.write_text("not a model")

        result = register_asset(
            model_id="bad-model",
            file_path=str(txt_file),
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert ".gguf" in parsed["error"].lower() or "invalid" in parsed["error"].lower()
