from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.models import GenerateResult
from aurarouter.savings.privacy import PrivacyAuditor, PrivacyStore
from aurarouter.savings.usage_store import UsageStore


def _make_config(models: dict, roles: dict) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return cfg


_OLLAMA_MODEL = {"provider": "ollama", "model_name": "test", "endpoint": "http://x"}
_GOOGLE_MODEL = {"provider": "google", "model_name": "gemini-2.0-flash", "api_key": "K"}


# ------------------------------------------------------------------
# Usage recording
# ------------------------------------------------------------------


def test_execute_records_usage(tmp_path, monkeypatch):
    """Mock provider, execute, verify UsageStore has 1 record with correct tokens."""
    config = _make_config(
        models={"m1": _OLLAMA_MODEL},
        roles={"coding": ["m1"]},
    )
    store = UsageStore(db_path=tmp_path / "usage.db")
    fabric = ComputeFabric(config, usage_store=store)

    fake_result = GenerateResult(text="hello", input_tokens=10, output_tokens=20)
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        result = fabric.execute("coding", "test prompt")

    assert result == "hello"
    records = store.query()
    assert len(records) == 1
    assert records[0].model_id == "m1"
    assert records[0].input_tokens == 10
    assert records[0].output_tokens == 20
    assert records[0].success is True


def test_execute_without_store(monkeypatch):
    """No store provided — execute works exactly as before."""
    config = _make_config(
        models={"m1": _OLLAMA_MODEL},
        roles={"coding": ["m1"]},
    )
    fabric = ComputeFabric(config)  # no savings components

    fake_result = GenerateResult(text="ok")
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        result = fabric.execute("coding", "prompt")

    assert result == "ok"


# ------------------------------------------------------------------
# Privacy audit
# ------------------------------------------------------------------


def test_execute_privacy_audit_logged(tmp_path, monkeypatch):
    """Cloud provider with PII → auto-reroutes to local fallback, event logged."""
    config = _make_config(
        models={"cloud": _GOOGLE_MODEL, "local": _OLLAMA_MODEL},
        roles={"coding": ["cloud", "local"]},
    )
    store = UsageStore(db_path=tmp_path / "usage.db")
    auditor = PrivacyAuditor()
    pstore = PrivacyStore(db_path=tmp_path / "privacy.db")
    fabric = ComputeFabric(
        config,
        usage_store=store,
        privacy_auditor=auditor,
        privacy_store=pstore,
    )

    fake_result = GenerateResult(text="response", input_tokens=5, output_tokens=15)
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        result = fabric.execute("coding", "contact user@example.com please")

    assert result == "response"
    events = pstore.query()
    assert len(events) == 1
    assert events[0]["match_count"] >= 1


def test_execute_privacy_audit_no_block(tmp_path, monkeypatch):
    """Privacy audit finds PII → cloud skipped, falls back to local."""
    config = _make_config(
        models={"cloud": _GOOGLE_MODEL, "local": _OLLAMA_MODEL},
        roles={"coding": ["cloud", "local"]},
    )
    auditor = PrivacyAuditor()
    pstore = PrivacyStore(db_path=tmp_path / "privacy.db")
    fabric = ComputeFabric(config, privacy_auditor=auditor, privacy_store=pstore)

    fake_result = GenerateResult(text="done", input_tokens=1, output_tokens=1)
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        result = fabric.execute("coding", "my ssn is 123-45-6789")

    assert result == "done"  # auto-rerouted to local


def test_execute_local_no_privacy_audit(tmp_path, monkeypatch):
    """Ollama (local) provider, prompt with email → NO privacy event."""
    config = _make_config(
        models={"local": _OLLAMA_MODEL},
        roles={"coding": ["local"]},
    )
    auditor = PrivacyAuditor()
    pstore = PrivacyStore(db_path=tmp_path / "privacy.db")
    fabric = ComputeFabric(config, privacy_auditor=auditor, privacy_store=pstore)

    fake_result = GenerateResult(text="local response")
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        result = fabric.execute("coding", "contact user@example.com please")

    assert result == "local response"
    events = pstore.query()
    assert len(events) == 0


# ------------------------------------------------------------------
# Callback compatibility
# ------------------------------------------------------------------


def test_callback_includes_tokens(monkeypatch):
    """Extended 6-arg callback receives input_tokens and output_tokens."""
    config = _make_config(
        models={"m1": _OLLAMA_MODEL},
        roles={"coding": ["m1"]},
    )
    fabric = ComputeFabric(config)

    captured = {}

    def ext_callback(role, model_id, success, elapsed, inp_tok, out_tok):
        captured["role"] = role
        captured["input_tokens"] = inp_tok
        captured["output_tokens"] = out_tok

    fake_result = GenerateResult(text="hi", input_tokens=7, output_tokens=13)
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        fabric.execute("coding", "prompt", on_model_tried=ext_callback)

    assert captured["input_tokens"] == 7
    assert captured["output_tokens"] == 13


def test_old_callback_still_works(monkeypatch):
    """4-arg callback still works without error."""
    config = _make_config(
        models={"m1": _OLLAMA_MODEL},
        roles={"coding": ["m1"]},
    )
    fabric = ComputeFabric(config)

    captured = {}

    def old_callback(role, model_id, success, elapsed):
        captured["role"] = role
        captured["success"] = success

    fake_result = GenerateResult(text="works")
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        fabric.execute("coding", "prompt", on_model_tried=old_callback)

    assert captured["role"] == "coding"
    assert captured["success"] is True


# ------------------------------------------------------------------
# Fallback records all attempts
# ------------------------------------------------------------------


def test_fallback_records_all_attempts(tmp_path, monkeypatch):
    """First model fails, second succeeds → 2 usage records."""
    config = _make_config(
        models={
            "m1": _OLLAMA_MODEL,
            "m2": {**_OLLAMA_MODEL, "model_name": "backup"},
        },
        roles={"coding": ["m1", "m2"]},
    )
    store = UsageStore(db_path=tmp_path / "usage.db")
    fabric = ComputeFabric(config, usage_store=store)

    call_count = 0

    def side_effect(prompt, json_mode=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("model down")
        return GenerateResult(text="recovered", input_tokens=3, output_tokens=5)

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        side_effect=side_effect,
    ):
        result = fabric.execute("coding", "prompt")

    assert result == "recovered"
    records = store.query()
    assert len(records) == 2
    assert records[0].success is False
    assert records[1].success is True
    assert records[1].input_tokens == 3
    assert records[1].output_tokens == 5
