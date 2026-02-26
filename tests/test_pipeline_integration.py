"""End-to-end pipeline integration tests.

Task Group C, TG2 — Tests the composed pipeline:
  user prompt → analyze_intent() → generate_plan() → fabric.execute()
with provider fallback, mocking only at the provider boundary.
"""

import json
from unittest.mock import patch, MagicMock, call

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import analyze_intent, generate_plan, TriageResult
from aurarouter.savings.models import GenerateResult
from aurarouter.savings.privacy import PrivacyAuditor, PrivacyStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fabric(models: dict, roles: dict, **extra_kwargs) -> ComputeFabric:
    """Build a fabric from inline config dicts with optional extras."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return ComputeFabric(cfg, **extra_kwargs)


# ---------------------------------------------------------------------------
# Task 2.1 — Simple Task Pipeline Test
# ---------------------------------------------------------------------------

class TestSimpleTaskPipeline:
    """Test the complete simple-task path end-to-end."""

    def test_simple_code_pipeline(self):
        fabric = _make_fabric(
            models={
                "m1": {"provider": "ollama", "model_name": "router-model", "endpoint": "http://x"},
            },
            roles={
                "router": ["m1"],
                "coding": ["m1"],
            },
        )

        # Track calls to distinguish router vs coding invocations
        call_log = []

        def mock_generate(prompt, json_mode=False):
            call_log.append(prompt)
            # First call is the router (intent classification)
            if "CLASSIFY intent" in prompt:
                return GenerateResult(
                    text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})
                )
            # Second call is the coding execution
            return GenerateResult(text="def add(a, b): return a + b")

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            side_effect=mock_generate,
        ):
            result = analyze_intent(fabric, "write an add function")
            assert isinstance(result, TriageResult)
            assert result.intent == "SIMPLE_CODE"
            assert result.complexity == 3

            code_output = fabric.execute("coding", "write an add function")
            assert code_output == "def add(a, b): return a + b"

        # Router called once for classification, coding called once for execution
        assert len(call_log) == 2
        assert "CLASSIFY intent" in call_log[0]
        assert "write an add function" in call_log[1]


# ---------------------------------------------------------------------------
# Task 2.2 — Complex Task Pipeline Test (Multi-Step)
# ---------------------------------------------------------------------------

class TestComplexTaskPipeline:
    """Test the complete multi-step path end-to-end."""

    def test_complex_reasoning_pipeline(self):
        fabric = _make_fabric(
            models={
                "m_router": {"provider": "ollama", "model_name": "router", "endpoint": "http://x"},
                "m_reasoning": {"provider": "ollama", "model_name": "reasoning", "endpoint": "http://y"},
                "m_coding": {"provider": "ollama", "model_name": "coding", "endpoint": "http://z"},
            },
            roles={
                "router": ["m_router"],
                "reasoning": ["m_reasoning"],
                "coding": ["m_coding"],
            },
        )

        invocations = []

        def mock_generate(prompt, json_mode=False):
            invocations.append(("generate", prompt[:60]))
            if "CLASSIFY intent" in prompt:
                return GenerateResult(
                    text=json.dumps({"intent": "COMPLEX_REASONING", "complexity": 8})
                )
            if "Lead Software Architect" in prompt:
                return GenerateResult(
                    text=json.dumps(["Create models.py", "Implement User class", "Add tests"])
                )
            # Coding calls
            return GenerateResult(text=f"# code for: {prompt[:30]}")

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            side_effect=mock_generate,
        ):
            # Step 1: Classify intent
            triage = analyze_intent(fabric, "build user management system")
            assert triage.intent == "COMPLEX_REASONING"
            assert triage.complexity == 8

            # Step 2: Generate plan
            plan = generate_plan(
                fabric, "build user management system", "Python web app"
            )
            assert plan == ["Create models.py", "Implement User class", "Add tests"]

            # Step 3: Execute each step
            results = []
            for step in plan:
                output = fabric.execute("coding", step)
                assert output is not None
                results.append(output)

        # Verify invocation order: router → reasoning → coding × 3
        assert len(invocations) == 5
        assert "CLASSIFY" in invocations[0][1]
        assert "Architect" in invocations[1][1]
        # Steps 2-4 are coding calls
        assert all("# code for:" in r for r in results)


# ---------------------------------------------------------------------------
# Task 2.3 — Fallback Chain Pipeline Test
# ---------------------------------------------------------------------------

class TestFallbackChainPipeline:
    """Test provider fallback within the pipeline."""

    def test_fallback_from_local_to_cloud(self):
        fabric = _make_fabric(
            models={
                "local_model": {"provider": "ollama", "model_name": "local", "endpoint": "http://x"},
                "cloud_model": {"provider": "ollama", "model_name": "cloud", "endpoint": "http://y"},
            },
            roles={"coding": ["local_model", "cloud_model"]},
        )

        callback_log = []

        def callback(role, model_id, success, elapsed):
            callback_log.append({"role": role, "model_id": model_id, "success": success})

        call_count = 0

        def mock_generate(prompt, json_mode=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("connection refused")
            return GenerateResult(text="fallback output")

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            side_effect=mock_generate,
        ):
            result = fabric.execute("coding", "test prompt", on_model_tried=callback)

        assert result == "fallback output"

        # Callback fired twice: first failure, then success
        assert len(callback_log) == 2
        assert callback_log[0]["model_id"] == "local_model"
        assert callback_log[0]["success"] is False
        assert callback_log[1]["model_id"] == "cloud_model"
        assert callback_log[1]["success"] is True


# ---------------------------------------------------------------------------
# Task 2.4 — Privacy Re-Route Pipeline Test
# ---------------------------------------------------------------------------

class TestPrivacyReRoutePipeline:
    """Test privacy-aware routing within the pipeline."""

    def test_pii_reroutes_from_cloud_to_local(self, tmp_path):
        privacy_store = PrivacyStore(db_path=tmp_path / "privacy.db")
        privacy_auditor = PrivacyAuditor()

        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "cloud_gemini": {
                    "provider": "google",
                    "model_name": "gemini-2.0-flash",
                    "api_key": "MOCK_KEY",
                },
                "local_qwen": {
                    "provider": "ollama",
                    "model_name": "qwen2.5-coder",
                    "endpoint": "http://localhost:11434/api/generate",
                },
            },
            "roles": {"coding": ["cloud_gemini", "local_qwen"]},
        }
        fabric = ComputeFabric(
            cfg,
            privacy_auditor=privacy_auditor,
            privacy_store=privacy_store,
        )

        callback_log = []

        def callback(role, model_id, success, elapsed):
            callback_log.append({"model_id": model_id, "success": success})

        # Mock providers — only Ollama should actually be called
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            return_value=GenerateResult(text="processed data safely"),
        ):
            result = fabric.execute(
                "coding",
                "Process user 123-45-6789 data",
                on_model_tried=callback,
            )

        assert result == "processed data safely"

        # cloud_gemini should be skipped (PII detected, cloud, no "private" tag)
        assert len(callback_log) == 2
        assert callback_log[0]["model_id"] == "cloud_gemini"
        assert callback_log[0]["success"] is False  # Skipped
        assert callback_log[1]["model_id"] == "local_qwen"
        assert callback_log[1]["success"] is True

        # Privacy event should be recorded
        events = privacy_store.query()
        assert len(events) >= 1
        assert events[0]["model_id"] == "cloud_gemini"
        assert "SSN" in events[0]["pattern_names"]
