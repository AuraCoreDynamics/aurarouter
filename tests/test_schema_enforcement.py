"""Tests for TG6: Intent-Aware Execution & Provider Schema Enforcement."""

import copy
from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import (
    ComputeFabric,
    MODIFICATIONS_SCHEMA,
    compile_modifications_schema,
)
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fabric(models: dict, roles: dict, **kwargs) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return ComputeFabric(cfg, **kwargs)


OLLAMA_MODEL = {
    "provider": "ollama",
    "model_name": "test-model",
    "endpoint": "http://localhost:11434/api/generate",
}

LLAMACPP_MODEL = {
    "provider": "llamacpp-server",
    "model_name": "test-model",
    "endpoint": "http://localhost:8080",
}

OPENAPI_MODEL = {
    "provider": "openapi",
    "model_name": "test-model",
    "endpoint": "http://localhost:8000/v1",
}


# ===================================================================
# T6.1 — MODIFICATIONS_SCHEMA constant
# ===================================================================

class TestModificationsSchema:
    """MODIFICATIONS_SCHEMA is well-formed."""

    def test_schema_is_dict(self):
        assert isinstance(MODIFICATIONS_SCHEMA, dict)

    def test_schema_type_is_object(self):
        assert MODIFICATIONS_SCHEMA["type"] == "object"

    def test_schema_requires_modifications(self):
        assert "modifications" in MODIFICATIONS_SCHEMA["required"]

    def test_modifications_is_array(self):
        mods = MODIFICATIONS_SCHEMA["properties"]["modifications"]
        assert mods["type"] == "array"

    def test_items_required_fields(self):
        items = MODIFICATIONS_SCHEMA["properties"]["modifications"]["items"]
        assert set(items["required"]) == {
            "file_path", "modification_type", "content", "language",
        }

    def test_modification_type_enum(self):
        items = MODIFICATIONS_SCHEMA["properties"]["modifications"]["items"]
        enum = items["properties"]["modification_type"]["enum"]
        assert "full_rewrite" in enum
        assert "unified_diff" in enum

    def test_schema_is_module_level_singleton(self):
        """Ensure the schema constant is not redefined elsewhere."""
        from aurarouter import fabric
        assert fabric.MODIFICATIONS_SCHEMA is MODIFICATIONS_SCHEMA


# ===================================================================
# T6.1 — compile_modifications_schema()
# ===================================================================

class TestCompileModificationsSchema:
    """Dynamic schema compiler."""

    def test_no_constraints_returns_base_schema(self):
        result = compile_modifications_schema(None)
        assert result is MODIFICATIONS_SCHEMA

    def test_empty_list_returns_base_schema(self):
        result = compile_modifications_schema([])
        assert result is MODIFICATIONS_SCHEMA

    def test_constraint_without_preferred_returns_base(self):
        """Constraints that don't mandate unified_diff don't narrow."""
        constraints = [{"path": "foo.py"}]
        result = compile_modifications_schema(constraints)
        # Should be a deep copy but with no allOf
        items = result["properties"]["modifications"]["items"]
        assert "allOf" not in items

    def test_single_diff_constraint_adds_allOf(self):
        constraints = [
            {"path": "src/main.py", "preferred_modification": "unified_diff"},
        ]
        result = compile_modifications_schema(constraints)
        items = result["properties"]["modifications"]["items"]
        assert "allOf" in items
        assert len(items["allOf"]) == 1
        cond = items["allOf"][0]
        assert cond["if"]["properties"]["file_path"]["const"] == "src/main.py"
        assert cond["then"]["properties"]["modification_type"]["enum"] == ["unified_diff"]

    def test_multiple_diff_constraints(self):
        constraints = [
            {"path": "a.py", "preferred_modification": "unified_diff"},
            {"path": "b.py", "preferred_modification": "unified_diff"},
        ]
        result = compile_modifications_schema(constraints)
        items = result["properties"]["modifications"]["items"]
        assert len(items["allOf"]) == 2

    def test_mixed_constraints(self):
        """Only files with preferred_modification='unified_diff' appear in allOf."""
        constraints = [
            {"path": "a.py", "preferred_modification": "unified_diff"},
            {"path": "b.py", "preferred_modification": "full_rewrite"},
            {"path": "c.py"},  # No preference
        ]
        result = compile_modifications_schema(constraints)
        items = result["properties"]["modifications"]["items"]
        assert len(items["allOf"]) == 1
        assert items["allOf"][0]["if"]["properties"]["file_path"]["const"] == "a.py"

    def test_does_not_mutate_base_schema(self):
        original = copy.deepcopy(MODIFICATIONS_SCHEMA)
        compile_modifications_schema([
            {"path": "x.py", "preferred_modification": "unified_diff"},
        ])
        assert MODIFICATIONS_SCHEMA == original


# ===================================================================
# T6.2 — execute() with options
# ===================================================================

class TestExecuteWithOptions:
    """ComputeFabric.execute() intent-aware schema enforcement."""

    def test_actionable_intent_sets_json_mode_and_schema(self):
        """edit_code intent should trigger json_mode and pass schema to provider."""
        fabric = _make_fabric(
            models={"m1": OLLAMA_MODEL},
            roles={"coding": ["m1"]},
        )
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text='{"modifications": []}'),
        ) as mock_gen:
            result = fabric.execute(
                "coding", "edit this file",
                options={"intent": "edit_code"},
            )
            mock_gen.assert_called_once()
            call_args = mock_gen.call_args
            # json_mode passed as kwarg
            assert call_args.kwargs.get("json_mode") is True or call_args[1].get("json_mode") is True
            # response_schema should be present and contain modifications
            schema = call_args.kwargs.get("response_schema") or call_args[1].get("response_schema")
            assert schema is not None
            assert "modifications" in schema["properties"]

    def test_generate_code_intent_also_actionable(self):
        fabric = _make_fabric(
            models={"m1": OLLAMA_MODEL},
            roles={"coding": ["m1"]},
        )
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text='{"modifications": []}'),
        ) as mock_gen:
            fabric.execute(
                "coding", "write a function",
                options={"intent": "generate_code"},
            )
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("json_mode") is True or call_args[1].get("json_mode") is True
            schema = call_args.kwargs.get("response_schema") or call_args[1].get("response_schema")
            assert schema is not None

    def test_chat_intent_no_schema(self):
        fabric = _make_fabric(
            models={"m1": OLLAMA_MODEL},
            roles={"coding": ["m1"]},
        )
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text="hello there"),
        ) as mock_gen:
            fabric.execute(
                "coding", "hi",
                options={"intent": "chat"},
            )
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("json_mode") is False or call_args[1].get("json_mode") is False
            # response_schema should NOT be passed (not in kwargs)
            assert "response_schema" not in (call_args.kwargs or call_args[1])

    def test_none_options_backward_compat(self):
        """execute() with no options arg still works as before."""
        fabric = _make_fabric(
            models={"m1": OLLAMA_MODEL},
            roles={"coding": ["m1"]},
        )
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text="result"),
        ) as mock_gen:
            result = fabric.execute("coding", "prompt")
            assert result is not None
            assert result.text == "result"
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("json_mode") is False or call_args[1].get("json_mode") is False
            # response_schema should NOT be in kwargs (None schema is not passed)
            assert "response_schema" not in (call_args.kwargs or call_args[1])

    def test_json_mode_true_without_options_preserved(self):
        """Existing json_mode=True without options still works."""
        fabric = _make_fabric(
            models={"m1": OLLAMA_MODEL},
            roles={"coding": ["m1"]},
        )
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text='{"key": "value"}'),
        ) as mock_gen:
            result = fabric.execute("coding", "prompt", json_mode=True)
            assert result is not None
            assert result.text == '{"key": "value"}'
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("json_mode") is True or call_args[1].get("json_mode") is True
            # response_schema should NOT be in kwargs
            assert "response_schema" not in (call_args.kwargs or call_args[1])

    def test_actionable_intent_with_file_constraints(self):
        """File constraints are threaded to compile_modifications_schema."""
        fabric = _make_fabric(
            models={"m1": OLLAMA_MODEL},
            roles={"coding": ["m1"]},
        )
        constraints = [
            {"path": "main.py", "preferred_modification": "unified_diff"},
        ]
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text='{"modifications": []}'),
        ) as mock_gen:
            fabric.execute(
                "coding", "edit main.py",
                options={"intent": "edit_code", "file_constraints": constraints},
            )
            call_args = mock_gen.call_args
            schema = call_args.kwargs.get("response_schema") or call_args[1].get("response_schema")
            items = schema["properties"]["modifications"]["items"]
            assert "allOf" in items
            assert items["allOf"][0]["if"]["properties"]["file_path"]["const"] == "main.py"

    def test_actionable_overrides_explicit_json_mode_false(self):
        """Even if json_mode=False is passed, actionable intent forces True."""
        fabric = _make_fabric(
            models={"m1": OLLAMA_MODEL},
            roles={"coding": ["m1"]},
        )
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text='{"modifications": []}'),
        ) as mock_gen:
            fabric.execute(
                "coding", "edit this",
                json_mode=False,
                options={"intent": "edit_code"},
            )
            call_args = mock_gen.call_args
            assert call_args.kwargs.get("json_mode") is True or call_args[1].get("json_mode") is True


# ===================================================================
# T6.4–T6.6 — Provider payload tests
# ===================================================================

class TestOllamaSchemaInjection:
    """Ollama provider injects schema as format value."""

    def test_schema_sets_format_to_schema_dict(self):
        from aurarouter.providers.ollama import OllamaProvider
        provider = OllamaProvider(OLLAMA_MODEL)
        schema = MODIFICATIONS_SCHEMA

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "response": '{"modifications": []}',
                "prompt_eval_count": 10,
                "eval_count": 20,
            }
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post = MagicMock(return_value=mock_resp)

            provider.generate("test", json_mode=True, response_schema=schema)
            call_kwargs = mock_client_cls.return_value.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
            assert payload["format"] is schema

    def test_json_mode_without_schema_uses_string(self):
        from aurarouter.providers.ollama import OllamaProvider
        provider = OllamaProvider(OLLAMA_MODEL)

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "response": '{}',
                "prompt_eval_count": 0,
                "eval_count": 0,
            }
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post = MagicMock(return_value=mock_resp)

            provider.generate("test", json_mode=True, response_schema=None)
            call_kwargs = mock_client_cls.return_value.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
            assert payload["format"] == "json"


class TestLlamaCppSchemaInjection:
    """llama.cpp server provider injects schema via response_format."""

    def test_schema_sets_response_format(self):
        from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
        provider = LlamaCppServerProvider(LLAMACPP_MODEL)
        schema = MODIFICATIONS_SCHEMA

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "content": '{"modifications": []}',
                "tokens_evaluated": 10,
                "tokens_predicted": 20,
            }
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post = MagicMock(return_value=mock_resp)

            provider.generate("test", json_mode=True, response_schema=schema)
            call_kwargs = mock_client_cls.return_value.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
            assert payload["response_format"] == {
                "type": "json_object",
                "schema": schema,
            }
            # json_schema key should NOT be set when response_format is used
            assert "json_schema" not in payload

    def test_json_mode_without_schema_uses_json_schema(self):
        from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
        provider = LlamaCppServerProvider(LLAMACPP_MODEL)

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "content": '{}',
                "tokens_evaluated": 0,
                "tokens_predicted": 0,
            }
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post = MagicMock(return_value=mock_resp)

            provider.generate("test", json_mode=True, response_schema=None)
            call_kwargs = mock_client_cls.return_value.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
            assert "json_schema" in payload
            assert "response_format" not in payload


class TestOpenAPISchemaInjection:
    """OpenAI-compatible provider injects schema via json_schema format."""

    def test_schema_sets_json_schema_format(self):
        from aurarouter.providers.openapi import OpenAPIProvider
        provider = OpenAPIProvider(OPENAPI_MODEL)
        schema = MODIFICATIONS_SCHEMA

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"modifications": []}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post = MagicMock(return_value=mock_resp)

            provider.generate("test", json_mode=True, response_schema=schema)
            call_kwargs = mock_client_cls.return_value.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
            rf = payload["response_format"]
            assert rf["type"] == "json_schema"
            assert rf["json_schema"]["name"] == "modifications"
            assert rf["json_schema"]["strict"] is True
            assert rf["json_schema"]["schema"] is schema

    def test_json_mode_without_schema_uses_json_object(self):
        from aurarouter.providers.openapi import OpenAPIProvider
        provider = OpenAPIProvider(OPENAPI_MODEL)

        with patch("httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{}'}}],
                "usage": {},
            }
            mock_client_cls.return_value.__enter__ = lambda s: s
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.post = MagicMock(return_value=mock_resp)

            provider.generate("test", json_mode=True, response_schema=None)
            call_kwargs = mock_client_cls.return_value.post.call_args
            payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs["json"]
            assert payload["response_format"] == {"type": "json_object"}
