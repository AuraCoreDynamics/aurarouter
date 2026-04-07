from unittest.mock import patch, MagicMock, call

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric, _ModelAttempt
from aurarouter.savings.models import GenerateResult


def _make_fabric(models: dict, roles: dict, **kwargs) -> ComputeFabric:
    """Build a fabric from inline config dicts."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return ComputeFabric(cfg, **kwargs)


def test_execute_returns_first_success():
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="hello world code"),
    ):
        result = fabric.execute("coding", "test prompt")
    assert result is not None
    assert result.text == "hello world code"


def test_execute_skips_empty_response():
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1", "m2"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=[
            GenerateResult(text=""),
            GenerateResult(text="valid result here"),
        ],
    ):
        result = fabric.execute("coding", "prompt")
    assert result is not None
    assert result.text == "valid result here"


def test_execute_returns_none_when_all_fail():
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=Exception("boom"),
    ):
        result = fabric.execute("coding", "prompt")
    assert result is None


def test_execute_unknown_role():
    fabric = _make_fabric(models={}, roles={})
    result = fabric.execute("nonexistent", "prompt")
    assert result is not None
    assert "ERROR" in result.text


# ------------------------------------------------------------------
# chain_override
# ------------------------------------------------------------------

def test_execute_with_chain_override():
    """chain_override should bypass the role's configured chain."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1"]},  # Only m1 in role chain
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="from m2"),
    ):
        # Override to use m2 instead
        result = fabric.execute("coding", "test", chain_override=["m2"])
    assert result is not None
    assert result.text == "from m2"


def test_execute_chain_override_none_uses_role():
    """chain_override=None should use the normal role chain."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="normal result"),
    ):
        result = fabric.execute("coding", "test", chain_override=None)
    assert result is not None
    assert result.text == "normal result"


# ------------------------------------------------------------------
# execute_all
# ------------------------------------------------------------------

def test_execute_all_collects_all():
    """execute_all should return results from every model, not just the first."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1", "m2"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=[
            GenerateResult(text="response1", input_tokens=10, output_tokens=20),
            GenerateResult(text="response2", input_tokens=15, output_tokens=25),
        ],
    ):
        results = fabric.execute_all("coding", "test prompt")

    assert len(results) == 2
    assert results[0]["model_id"] == "m1"
    assert results[0]["success"] is True
    assert results[0]["text"] == "response1"
    assert results[1]["model_id"] == "m2"
    assert results[1]["success"] is True
    assert results[1]["text"] == "response2"


def test_execute_all_with_explicit_model_ids():
    """execute_all should accept explicit model_ids."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        roles={"coding": ["m1", "m2"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="only m2"),
    ):
        results = fabric.execute_all("coding", "test", model_ids=["m2"])

    assert len(results) == 1
    assert results[0]["model_id"] == "m2"


def test_execute_all_handles_failures():
    """execute_all should record failures without stopping."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=Exception("connection refused"),
    ):
        results = fabric.execute_all("coding", "test")

    assert len(results) == 1
    assert results[0]["success"] is False
    assert "ERROR" in results[0]["text"]


# ------------------------------------------------------------------
# _try_model edge cases
# ------------------------------------------------------------------

def test_try_model_budget_blocked_returns_skipped():
    """Budget-blocked cloud models return skipped=True."""
    budget_mgr = MagicMock()
    budget_mgr.check_budget.return_value = MagicMock(
        allowed=False, reason="daily limit exceeded",
    )

    fabric = _make_fabric(
        models={
            "cloud1": {
                "provider": "google",
                "model_name": "gemini-pro",
                "hosting_tier": "cloud",
            },
        },
        roles={"coding": ["cloud1"]},
        budget_manager=budget_mgr,
    )

    attempt = fabric._try_model("coding", "cloud1", lambda p: None)
    assert attempt.skipped is True
    assert attempt.success is False
    assert attempt.result is None
    assert "daily limit exceeded" in attempt.error


def test_try_model_privacy_blocked_returns_skipped():
    """Privacy-blocked cloud models (PII detected, not tagged 'private') return skipped=True."""
    privacy_auditor = MagicMock()
    # Return a non-None event indicating PII was detected
    mock_event = MagicMock()
    mock_event.matches = [MagicMock()]  # One match
    privacy_auditor.audit.return_value = mock_event

    fabric = _make_fabric(
        models={
            "cloud1": {
                "provider": "google",
                "model_name": "gemini-pro",
                "hosting_tier": "cloud",
            },
        },
        roles={"coding": ["cloud1"]},
        privacy_auditor=privacy_auditor,
    )

    with patch(
        "aurarouter.providers.get_provider",
        return_value=MagicMock(),
    ):
        attempt = fabric._try_model(
            "coding", "cloud1", lambda p: None,
            audit_text="my SSN is 123-45-6789",
        )

    assert attempt.skipped is True
    assert attempt.success is False
    assert "PII detected" in attempt.error


def test_try_model_provider_cache_populated():
    """Provider cache is populated after first _try_model call."""
    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )
    assert "m1" not in fabric._provider_cache

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="ok"),
    ):
        attempt = fabric._try_model(
            "coding", "m1",
            lambda p: p.generate_with_usage("test"),
        )

    assert attempt.success is True
    assert "m1" in fabric._provider_cache


def test_try_model_callback_fired_on_success():
    """on_model_tried callback fires with success=True on success."""
    callback = MagicMock()

    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="result", input_tokens=5, output_tokens=10),
    ):
        fabric._try_model(
            "coding", "m1",
            lambda p: p.generate_with_usage("test"),
            on_model_tried=callback,
        )

    callback.assert_called_once()
    call_args = callback.call_args[0]
    assert call_args[0] == "coding"   # role
    assert call_args[1] == "m1"       # model_id
    assert call_args[2] is True       # success
    assert call_args[3] > 0           # elapsed > 0


def test_try_model_callback_fired_on_failure():
    """on_model_tried callback fires with success=False on provider error."""
    callback = MagicMock()

    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=Exception("connection refused"),
    ):
        fabric._try_model(
            "coding", "m1",
            lambda p: p.generate_with_usage("test"),
            on_model_tried=callback,
        )

    callback.assert_called_once()
    call_args = callback.call_args[0]
    assert call_args[0] == "coding"
    assert call_args[1] == "m1"
    assert call_args[2] is False  # success


def test_try_model_usage_recorded_on_success():
    """Usage store records a successful attempt."""
    usage_store = MagicMock()

    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
        usage_store=usage_store,
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=GenerateResult(text="ok", input_tokens=10, output_tokens=20),
    ):
        fabric._try_model(
            "coding", "m1",
            lambda p: p.generate_with_usage("test"),
            intent="test_intent",
        )

    usage_store.record.assert_called_once()
    record = usage_store.record.call_args[0][0]
    assert record.model_id == "m1"
    assert record.success is True
    assert record.input_tokens == 10
    assert record.output_tokens == 20
    assert record.intent == "test_intent"


def test_try_model_usage_recorded_on_failure():
    """Usage store records a failed attempt with zero tokens."""
    usage_store = MagicMock()

    fabric = _make_fabric(
        models={
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        roles={"coding": ["m1"]},
        usage_store=usage_store,
    )

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=RuntimeError("timeout"),
    ):
        fabric._try_model(
            "coding", "m1",
            lambda p: p.generate_with_usage("test"),
        )

    usage_store.record.assert_called_once()
    record = usage_store.record.call_args[0][0]
    assert record.model_id == "m1"
    assert record.success is False
    assert record.input_tokens == 0
    assert record.output_tokens == 0


def test_try_model_missing_config_returns_skipped():
    """_try_model with a non-existent model_id returns skipped=True."""
    fabric = _make_fabric(models={}, roles={})
    attempt = fabric._try_model("coding", "nonexistent", lambda p: None)
    assert attempt.skipped is True
    assert attempt.success is False


def test_try_model_callback_fired_on_budget_skip():
    """on_model_tried callback fires with success=False on budget skip."""
    budget_mgr = MagicMock()
    budget_mgr.check_budget.return_value = MagicMock(
        allowed=False, reason="monthly limit exceeded",
    )
    callback = MagicMock()

    fabric = _make_fabric(
        models={
            "cloud1": {
                "provider": "google",
                "model_name": "gemini-pro",
                "hosting_tier": "cloud",
            },
        },
        roles={"coding": ["cloud1"]},
        budget_manager=budget_mgr,
    )

    fabric._try_model(
        "coding", "cloud1", lambda p: None,
        on_model_tried=callback,
    )

    callback.assert_called_once()
    call_args = callback.call_args[0]
    assert call_args[0] == "coding"
    assert call_args[1] == "cloud1"
    assert call_args[2] is False  # success


# ------------------------------------------------------------------
# TG5 / KI-004: Replica-count header and 2429 retry-after (T5.3 / T5.4)
# ------------------------------------------------------------------

def _make_xlm_fabric(replica_count: int = 1, extra_config: dict | None = None) -> ComputeFabric:
    """Build a fabric with XLM augmentation enabled via injected xlm_client."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {},
        "roles": {},
        "xlm": {"endpoint": "http://xlm-mock", "augmentation": True, "usage_reporting": True},
        "replica_count": replica_count,
        **(extra_config or {}),
    }
    return ComputeFabric(cfg)


def test_augment_prompt_injects_replica_count_header():
    """X-AuraCore-Replica-Count header must be present on every XLM call."""
    fabric = _make_xlm_fabric(replica_count=3)

    captured_headers: list[dict] = []

    mock_client = MagicMock()
    mock_client.last_response_headers = {}
    mock_client.call_tool.side_effect = lambda tool, headers=None, **kw: (
        captured_headers.append(headers or {}) or {"augmented_prompt": "ctx: " + kw.get("prompt", "")}
    )

    fabric._xlm_client = mock_client

    with patch.object(fabric._config, "is_xlm_augmentation_enabled", return_value=True), \
         patch.object(fabric._config, "get_xlm_endpoint", return_value="http://xlm-mock"):
        fabric._augment_prompt("hello", "coding")

    assert captured_headers, "call_tool should have been called at least once"
    for h in captured_headers:
        assert h.get("X-AuraCore-Replica-Count") == "3", (
            f"Expected header value '3', got {h.get('X-AuraCore-Replica-Count')!r}"
        )


def test_augment_prompt_2429_sleeps_and_retries_once():
    """When XLM returns 2429 with Retry-After, fabric sleeps ~that long then retries exactly once."""
    import time as _time_mod

    fabric = _make_xlm_fabric(replica_count=1)

    retry_after_secs = 2

    call_count = 0

    def _mock_call_tool(tool, headers=None, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"error_code": 2429, "retry_after_seconds": retry_after_secs}
        return {"augmented_prompt": "retried-ok"}

    mock_client = MagicMock()
    mock_client.last_response_headers = {"Retry-After": str(retry_after_secs)}
    mock_client.call_tool = _mock_call_tool

    fabric._xlm_client = mock_client

    slept_for: list[float] = []

    with patch.object(fabric._config, "is_xlm_augmentation_enabled", return_value=True), \
         patch.object(fabric._config, "get_xlm_endpoint", return_value="http://xlm-mock"), \
         patch("aurarouter.fabric.time.sleep", side_effect=lambda s: slept_for.append(s)):
        result = fabric._augment_prompt("question", "coding")

    assert call_count == 2, f"Expected exactly 2 call_tool calls (1 initial + 1 retry), got {call_count}"
    assert result == "retried-ok", f"Expected augmented result after retry, got {result!r}"
    assert len(slept_for) == 1, f"Expected exactly one sleep, got {slept_for}"
    assert slept_for[0] >= retry_after_secs, (
        f"Sleep ({slept_for[0]:.2f}s) must be >= Retry-After ({retry_after_secs}s)"
    )
