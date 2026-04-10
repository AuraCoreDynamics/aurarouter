"""Tests for ONNXVectorAnalyzer (TG2).

Tests graceful degradation, model resolution, protocol conformance,
and integration with internal package resources.

TG2 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.analyzer_protocol import PromptAnalyzer
from aurarouter.analyzers.onnx_vector import ONNXVectorAnalyzer
from aurarouter.intent_registry import IntentRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_registry() -> IntentRegistry:
    return IntentRegistry()


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """ONNXVectorAnalyzer must not crash when dependencies are missing."""

    def test_onnx_analyzer_abstains_without_onnxruntime(
        self, empty_registry: IntentRegistry
    ) -> None:
        """If onnxruntime not installed, supports() → False."""
        import sys
        # Provide a fake model path so it doesn't fail on model resolution
        with patch.object(
            ONNXVectorAnalyzer, "_resolve_companion_model_path", return_value="/fake/model.onnx"
        ), patch.dict(sys.modules, {"onnxruntime": None}):
            analyzer = ONNXVectorAnalyzer(
                intent_registry=empty_registry,
                model_path="/fake/model.onnx",
            )
            # _try_load() will fail because onnxruntime import raises
            assert not analyzer.supports("test prompt")

    def test_onnx_analyzer_abstains_when_model_file_missing(
        self, empty_registry: IntentRegistry
    ) -> None:
        """If model file doesn't exist, supports() → False."""
        analyzer = ONNXVectorAnalyzer(
            intent_registry=empty_registry,
            model_path="/nonexistent/path/model.onnx",
        )
        assert not analyzer.supports("test prompt")


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------


class TestModelPathResolution:
    def test_onnx_analyzer_resolves_model_from_resources(
        self, empty_registry: IntentRegistry
    ) -> None:
        """_resolve_companion_model_path() uses importlib.resources."""
        with patch("importlib.resources.files") as mock_files:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_files.return_value.__truediv__.return_value = mock_path
            
            path = ONNXVectorAnalyzer._resolve_companion_model_path()
            assert path is not None
            mock_files.assert_called_with("aurarouter.resources.onnx")

    def test_model_path_override_bypasses_resource_resolution(
        self, empty_registry: IntentRegistry
    ) -> None:
        """Explicit model_path arg supersedes resource resolution."""
        with patch.object(
            ONNXVectorAnalyzer, "_resolve_companion_model_path", return_value="/resource/path.onnx"
        ) as mock_resolve:
            analyzer = ONNXVectorAnalyzer(
                intent_registry=empty_registry,
                model_path="/custom/path.onnx",
            )
            # Constructor should not call _resolve_companion_model_path when override given
            mock_resolve.assert_not_called()
            assert analyzer._model_path == "/custom/path.onnx"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_conforms_to_protocol(self, empty_registry: IntentRegistry) -> None:
        analyzer = ONNXVectorAnalyzer(
            intent_registry=empty_registry,
            model_path="/fake/path.onnx",
        )
        assert isinstance(analyzer, PromptAnalyzer)

    def test_analyzer_id(self, empty_registry: IntentRegistry) -> None:
        analyzer = ONNXVectorAnalyzer(intent_registry=empty_registry)
        assert analyzer.analyzer_id == "onnx-vector"

    def test_priority_100(self, empty_registry: IntentRegistry) -> None:
        analyzer = ONNXVectorAnalyzer(intent_registry=empty_registry)
        assert analyzer.priority == 100


# ---------------------------------------------------------------------------
# Complexity sentinel
# ---------------------------------------------------------------------------


class TestComplexitySentinel:
    def test_onnx_analyzer_returns_zero_complexity_sentinel(
        self, empty_registry: IntentRegistry
    ) -> None:
        """When ONNX successfully classifies, result.complexity_score must be 0 (sentinel)."""
        # Build a mock session that returns a dummy embedding
        import numpy as np

        mock_session = MagicMock()
        # Return a plausible last_hidden_state output: [batch=1, seq=5, hidden=384]
        mock_session.run.return_value = [np.random.rand(1, 5, 384).astype(np.float32)]

        analyzer = ONNXVectorAnalyzer(
            intent_registry=empty_registry,
            model_path="/fake/path.onnx",
        )
        analyzer._session = mock_session
        analyzer._loaded = True

        # Register a fake intent to get some similarity
        from aurarouter.intent_registry import IntentDefinition
        empty_registry.register(IntentDefinition(
            name="TEST_INTENT",
            description="A test intent for unit tests",
            target_role="coding",
            source="test",
        ))
        analyzer._build_intent_matrix()

        if analyzer._intent_matrix is not None:
            result = analyzer.analyze("test prompt")
            if result is not None:
                assert result.complexity_score == 0, \
                    f"Expected complexity_score=0 (sentinel), got {result.complexity_score}"


# ---------------------------------------------------------------------------
# Resource structure
# ---------------------------------------------------------------------------


class TestResourceStructure:
    """Verify the internal resource layout."""

    def test_resource_files_exist(self) -> None:
        import pathlib
        pkg_root = pathlib.Path(__file__).parent.parent / "src" / "aurarouter"
        assert (pkg_root / "resources" / "onnx" / "tokenizer.json").exists()
        assert (pkg_root / "resources" / "onnx" / "metadata.py").exists()
