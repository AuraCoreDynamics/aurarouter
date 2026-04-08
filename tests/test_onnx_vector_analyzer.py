"""Tests for ONNXVectorAnalyzer (TG2).

Tests graceful degradation, model resolution, protocol conformance,
and integration with the companion sidecar package.

TG2 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

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

    def test_onnx_analyzer_abstains_without_companion_package(
        self, empty_registry: IntentRegistry
    ) -> None:
        """If aurarouter_onnx is not installed, supports() → False, analyze() → None."""
        with patch.dict("sys.modules", {"aurarouter_onnx": None}):
            analyzer = ONNXVectorAnalyzer(
                intent_registry=empty_registry,
                model_path=None,
            )
            # supports() must return False
            assert not analyzer.supports("hello world")
            # analyze() must return None gracefully
            result = analyzer.analyze("hello world")
            assert result is None

    def test_onnx_analyzer_abstains_without_onnxruntime(
        self, empty_registry: IntentRegistry
    ) -> None:
        """If onnxruntime not installed, supports() → False."""
        import sys
        # Provide a fake model path so companion package isn't needed
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
    def test_onnx_analyzer_resolves_model_from_companion_package(
        self, empty_registry: IntentRegistry
    ) -> None:
        """_resolve_companion_model_path() calls aurarouter_onnx.get_model_path()."""
        mock_pkg = MagicMock()
        mock_pkg.get_model_path.return_value = "/path/to/sentence_encoder.onnx"
        mock_pkg.get_tokenizer_path.return_value = "/path/to/tokenizer.json"

        with patch.dict("sys.modules", {"aurarouter_onnx": mock_pkg}):
            path = ONNXVectorAnalyzer._resolve_companion_model_path()
            assert path == "/path/to/sentence_encoder.onnx"
            mock_pkg.get_model_path.assert_called_once()

    def test_model_path_override_bypasses_companion_package(
        self, empty_registry: IntentRegistry
    ) -> None:
        """Explicit model_path arg supersedes companion package resolution."""
        with patch.object(
            ONNXVectorAnalyzer, "_resolve_companion_model_path", return_value="/companion/path.onnx"
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
# Companion package structure
# ---------------------------------------------------------------------------


class TestCompanionPackageStructure:
    """Verify the companion package layout."""

    def test_companion_package_files_exist(self) -> None:
        import pathlib
        pkg_root = pathlib.Path(__file__).parent.parent.parent / "aurarouter-onnx"
        assert (pkg_root / "pyproject.toml").exists()
        assert (pkg_root / "src" / "aurarouter_onnx" / "__init__.py").exists()
        assert (pkg_root / "src" / "aurarouter_onnx" / "metadata.py").exists()
        assert (pkg_root / "src" / "aurarouter_onnx" / "model").is_dir()

    def test_companion_init_exports_expected_functions(self) -> None:
        """The companion package's __init__.py must export get_model_path and get_tokenizer_path."""
        import importlib.util
        import pathlib
        init_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "aurarouter-onnx" / "src" / "aurarouter_onnx" / "__init__.py"
        )
        spec = importlib.util.spec_from_file_location("aurarouter_onnx_test", init_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        assert callable(getattr(mod, "get_model_path", None))
        assert callable(getattr(mod, "get_tokenizer_path", None))

    def test_companion_metadata_has_expected_fields(self) -> None:
        import importlib.util
        import pathlib
        meta_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "aurarouter-onnx" / "src" / "aurarouter_onnx" / "metadata.py"
        )
        spec = importlib.util.spec_from_file_location("aurarouter_onnx_meta", meta_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        assert "package_name" in mod.METADATA
        assert "embedding_dim" in mod.METADATA
        assert mod.METADATA["embedding_dim"] == 384
