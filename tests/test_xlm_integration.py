"""Tests for XLM integration: RAG prompt augmentation and usage reporting."""

import threading
import time
from unittest.mock import MagicMock, patch, call

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric


def _make_config(xlm_config=None, models=None, roles=None):
    """Build a ConfigLoader with optional XLM config."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": models or {},
        "roles": roles or {},
    }
    if xlm_config is not None:
        cfg.config["xlm"] = xlm_config
    return cfg


def _make_fabric(xlm_config=None, models=None, roles=None, xlm_client=None):
    """Build a ComputeFabric with optional XLM config and client."""
    cfg = _make_config(xlm_config, models, roles)
    return ComputeFabric(cfg, xlm_client=xlm_client)


# ======================================================================
# T3.1: ConfigLoader XLM accessors
# ======================================================================

class TestConfigXlmAccessors:

    def test_get_xlm_config_empty_when_missing(self):
        cfg = _make_config()
        assert cfg.get_xlm_config() == {}

    def test_get_xlm_config_returns_section(self):
        cfg = _make_config(xlm_config={"endpoint": "http://xlm:8080"})
        assert cfg.get_xlm_config()["endpoint"] == "http://xlm:8080"

    def test_augmentation_disabled_by_default(self):
        cfg = _make_config()
        assert cfg.is_xlm_augmentation_enabled() is False

    def test_augmentation_enabled_when_set(self):
        cfg = _make_config(xlm_config={
            "features": {"prompt_augmentation": True}
        })
        assert cfg.is_xlm_augmentation_enabled() is True

    def test_augmentation_disabled_when_explicitly_false(self):
        cfg = _make_config(xlm_config={
            "features": {"prompt_augmentation": False}
        })
        assert cfg.is_xlm_augmentation_enabled() is False

    def test_usage_reporting_disabled_by_default(self):
        cfg = _make_config()
        assert cfg.is_xlm_usage_reporting_enabled() is False

    def test_usage_reporting_enabled_when_set(self):
        cfg = _make_config(xlm_config={
            "features": {"usage_reporting": True}
        })
        assert cfg.is_xlm_usage_reporting_enabled() is True

    def test_get_xlm_endpoint_empty_when_missing(self):
        cfg = _make_config()
        assert cfg.get_xlm_endpoint() == ""

    def test_get_xlm_endpoint_returns_url(self):
        cfg = _make_config(xlm_config={"endpoint": "http://xlm:9090"})
        assert cfg.get_xlm_endpoint() == "http://xlm:9090"


# ======================================================================
# T3.2: Prompt augmentation hook
# ======================================================================

class TestPromptAugmentation:

    def test_augmentation_skipped_when_config_disabled(self):
        """No XLM config means augmentation is skipped entirely."""
        fabric = _make_fabric()
        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "original prompt"

    def test_augmentation_skipped_when_no_endpoint(self):
        """Augmentation enabled but no endpoint returns original prompt."""
        fabric = _make_fabric(xlm_config={
            "features": {"prompt_augmentation": True},
        })
        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "original prompt"

    def test_augmentation_calls_auraxlm_query_dict_response(self):
        """When enabled with endpoint, calls auraxlm.query and uses dict result."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "augmented_prompt": "RAG context here. original prompt"
        }

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
            xlm_client=mock_client,
        )

        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "RAG context here. original prompt"
        mock_client.call_tool.assert_called_once_with(
            "auraxlm.query", prompt="original prompt", role="coding",
        )

    def test_augmentation_calls_auraxlm_query_string_response(self):
        """When auraxlm.query returns a plain string, use it."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = "augmented string result"

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
            xlm_client=mock_client,
        )

        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "augmented string result"

    def test_augmentation_returns_original_on_empty_string_response(self):
        """Empty string from auraxlm.query returns original prompt."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = "   "

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
            xlm_client=mock_client,
        )

        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "original prompt"

    def test_augmentation_returns_original_on_exception(self):
        """Any exception in augmentation returns original prompt."""
        mock_client = MagicMock()
        mock_client.call_tool.side_effect = RuntimeError("connection refused")

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
            xlm_client=mock_client,
        )

        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "original prompt"

    def test_augmentation_returns_original_on_dict_without_key(self):
        """Dict response without augmented_prompt key returns original."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"other_key": "value"}

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
            xlm_client=mock_client,
        )

        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "original prompt"

    def test_augmentation_returns_original_on_none_result(self):
        """None result from call_tool returns original prompt."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = None

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
            xlm_client=mock_client,
        )

        result = fabric._augment_prompt("original prompt", "coding")
        assert result == "original prompt"

    def test_augmentation_called_in_execute(self):
        """_augment_prompt is invoked at the top of execute()."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"augmented_prompt": "augmented!"}

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
            models={
                "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            },
            roles={"coding": ["m1"]},
            xlm_client=mock_client,
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            return_value="result text",
        ) as mock_gen:
            fabric.execute("coding", "original prompt")
            # The provider should have received the augmented prompt
            mock_gen.assert_called_once_with("augmented!", json_mode=False)

    def test_augmentation_without_xlm_client_connects_new(self):
        """Without injected xlm_client, creates GridMcpClient and calls connect."""
        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
        )

        mock_client_instance = MagicMock()
        mock_client_instance.connect.return_value = False

        with patch(
            "aurarouter.mcp_client.client.GridMcpClient",
            return_value=mock_client_instance,
        ):
            result = fabric._augment_prompt("original", "coding")

        assert result == "original"
        mock_client_instance.connect.assert_called_once()

    def test_augmentation_without_xlm_client_success(self):
        """Without injected xlm_client, creates GridMcpClient that succeeds."""
        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"prompt_augmentation": True},
            },
        )

        mock_client_instance = MagicMock()
        mock_client_instance.connect.return_value = True
        mock_client_instance.call_tool.return_value = {"augmented_prompt": "rag + original"}

        with patch(
            "aurarouter.mcp_client.client.GridMcpClient",
            return_value=mock_client_instance,
        ):
            result = fabric._augment_prompt("original", "coding")

        assert result == "rag + original"
        mock_client_instance.call_tool.assert_called_once_with(
            "auraxlm.query", prompt="original", role="coding",
        )


# ======================================================================
# T3.3: Usage reporting (fire-and-forget)
# ======================================================================

class TestUsageReporting:

    def test_usage_reporting_skipped_when_config_disabled(self):
        """No XLM config means _report_usage is a no-op."""
        fabric = _make_fabric()
        # Should not raise or start any thread
        with patch("threading.Thread") as mock_thread:
            fabric._report_usage("coding", "m1", True, 1.5)
            mock_thread.assert_not_called()

    def test_usage_reporting_skipped_when_no_endpoint(self):
        """Usage reporting enabled but no endpoint is a no-op."""
        fabric = _make_fabric(xlm_config={
            "features": {"usage_reporting": True},
        })
        with patch("threading.Thread") as mock_thread:
            fabric._report_usage("coding", "m1", True, 1.5)
            mock_thread.assert_not_called()

    def test_usage_reporting_spawns_thread(self):
        """When enabled, _report_usage spawns a daemon thread."""
        mock_client = MagicMock()

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"usage_reporting": True},
            },
            xlm_client=mock_client,
        )

        with patch("aurarouter.fabric.threading.Thread") as mock_thread_cls:
            mock_thread_instance = MagicMock()
            mock_thread_cls.return_value = mock_thread_instance

            fabric._report_usage("coding", "m1", True, 1.5, 100, 200)

            mock_thread_cls.assert_called_once()
            assert mock_thread_cls.call_args[1]["daemon"] is True
            mock_thread_instance.start.assert_called_once()

    def test_usage_reporting_calls_auraxlm_usage(self):
        """The background thread calls auraxlm.usage with correct args."""
        mock_client = MagicMock()

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"usage_reporting": True},
            },
            xlm_client=mock_client,
        )

        # Call _report_usage and run the thread function synchronously
        fabric._report_usage("coding", "m1", True, 1.5, 100, 200)
        # Wait briefly for daemon thread
        time.sleep(0.1)

        mock_client.call_tool.assert_called_with(
            "auraxlm.usage",
            model_id="m1",
            role="coding",
            success=True,
            elapsed_seconds=1.5,
            input_tokens=100,
            output_tokens=200,
        )

    def test_usage_reporting_never_raises(self):
        """Even if the client throws, _report_usage never raises."""
        mock_client = MagicMock()
        mock_client.call_tool.side_effect = RuntimeError("network error")

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"usage_reporting": True},
            },
            xlm_client=mock_client,
        )

        # Should not raise
        fabric._report_usage("coding", "m1", False, 0.5)
        # Wait for thread to finish
        time.sleep(0.1)

    def test_usage_reporting_called_in_execute_on_success(self):
        """_report_usage is called after a successful model attempt in execute()."""
        mock_client = MagicMock()

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"usage_reporting": True},
            },
            models={
                "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            },
            roles={"coding": ["m1"]},
            xlm_client=mock_client,
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            return_value="result text",
        ):
            with patch.object(fabric, "_report_usage") as mock_report:
                fabric.execute("coding", "test prompt")
                mock_report.assert_called_once()
                args = mock_report.call_args[0]
                assert args[0] == "coding"  # role
                assert args[1] == "m1"      # model_id
                assert args[2] is True      # success

    def test_usage_reporting_called_in_execute_on_failure(self):
        """_report_usage is called after a failed model attempt in execute()."""
        mock_client = MagicMock()

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"usage_reporting": True},
            },
            models={
                "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            },
            roles={"coding": ["m1"]},
            xlm_client=mock_client,
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            side_effect=Exception("boom"),
        ):
            with patch.object(fabric, "_report_usage") as mock_report:
                fabric.execute("coding", "test prompt")
                mock_report.assert_called_once()
                args = mock_report.call_args[0]
                assert args[0] == "coding"  # role
                assert args[1] == "m1"      # model_id
                assert args[2] is False     # success

    def test_usage_reporting_without_xlm_client_creates_new(self):
        """Without injected client, _report_usage creates a GridMcpClient."""
        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {"usage_reporting": True},
            },
        )

        mock_client_instance = MagicMock()
        mock_client_instance.connect.return_value = True

        with patch(
            "aurarouter.mcp_client.client.GridMcpClient",
            return_value=mock_client_instance,
        ):
            fabric._report_usage("coding", "m1", True, 1.0)
            time.sleep(0.2)

        mock_client_instance.connect.assert_called_once()
        mock_client_instance.call_tool.assert_called_once()


# ======================================================================
# T3.4: Constructor accepts xlm_client
# ======================================================================

class TestConstructorXlmClient:

    def test_constructor_accepts_xlm_client(self):
        """ComputeFabric accepts optional xlm_client parameter."""
        mock_client = MagicMock()
        cfg = _make_config()
        fabric = ComputeFabric(cfg, xlm_client=mock_client)
        assert fabric._xlm_client is mock_client

    def test_constructor_xlm_client_defaults_to_none(self):
        """xlm_client defaults to None when not provided."""
        cfg = _make_config()
        fabric = ComputeFabric(cfg)
        assert fabric._xlm_client is None

    def test_constructor_accepts_kwargs(self):
        """Constructor accepts extra kwargs (e.g., savings components) without error."""
        cfg = _make_config()
        fabric = ComputeFabric(cfg, usage_store=MagicMock(), budget_manager=MagicMock())
        # Should not raise


# ======================================================================
# End-to-end: both features together
# ======================================================================

class TestBothFeaturesGraceful:

    def test_both_disabled_by_default_no_side_effects(self):
        """With no XLM config, execute works normally without any XLM calls."""
        fabric = _make_fabric(
            models={
                "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            },
            roles={"coding": ["m1"]},
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            return_value="result",
        ):
            result = fabric.execute("coding", "test prompt")
        assert result == "result"

    def test_both_features_enabled_augmentation_fails_gracefully(self):
        """If augmentation fails, execute still works and usage is reported."""
        mock_client = MagicMock()
        # augmentation call fails
        mock_client.call_tool.side_effect = [
            RuntimeError("augmentation down"),  # augment call
        ]

        fabric = _make_fabric(
            xlm_config={
                "endpoint": "http://xlm:8080",
                "features": {
                    "prompt_augmentation": True,
                    "usage_reporting": True,
                },
            },
            models={
                "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
            },
            roles={"coding": ["m1"]},
            xlm_client=mock_client,
        )

        # Reset side_effect after augmentation — usage call should work
        def call_tool_side_effect(tool_name, **kwargs):
            if tool_name == "auraxlm.query":
                raise RuntimeError("augmentation down")
            return None  # usage call succeeds

        mock_client.call_tool.side_effect = call_tool_side_effect

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            return_value="result",
        ):
            result = fabric.execute("coding", "test prompt")

        # Execute should still succeed with original prompt
        assert result == "result"
