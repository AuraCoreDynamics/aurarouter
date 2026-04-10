"""Tests for ONNXProvider."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock modules globally for the test session
mock_ort = MagicMock()
mock_np = MagicMock()
mock_tokenizers = MagicMock()

# A very simple mock class that behaves like a numpy array for our needs
class SimpleMockArray:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data),) if not isinstance(data[0], list) else (len(data), len(data[0]))
    
    def __getitem__(self, idx):
        return SimpleMockArray(self.data[idx]) if isinstance(self.data[idx], list) else self.data[idx]
    
    def __sub__(self, other):
        val = other.data if isinstance(other, SimpleMockArray) else other
        return SimpleMockArray([i - val for i in self.data])
    
    def __truediv__(self, other):
        val = other.data if isinstance(other, SimpleMockArray) else other
        if val == 0: val = 1e-9
        return SimpleMockArray([i / val for i in self.data])
    
    def sum(self, axis=None):
        return sum(self.data)
    
    def tolist(self):
        return self.data
    
    def __len__(self):
        return len(self.data)

def mock_array_fn(data, dtype=None):
    if isinstance(data, SimpleMockArray):
        return data
    return SimpleMockArray(data)

mock_np.array.side_effect = mock_array_fn
mock_np.exp.side_effect = lambda x: SimpleMockArray([2.718**i for i in x.data]) if isinstance(x, SimpleMockArray) else 2.718**x
mock_np.max.side_effect = lambda x: max(x.data)
mock_np.argmax.side_effect = lambda x: x.data.index(max(x.data))
mock_np.mean.side_effect = lambda x, axis=None: SimpleMockArray([sum(i)/len(i) for i in zip(*x.data)]) if axis==0 else sum(x.data)/len(x.data)

# We must keep these in sys.modules for the whole test
sys.modules["onnxruntime"] = mock_ort
sys.modules["numpy"] = mock_np
sys.modules["tokenizers"] = mock_tokenizers

from aurarouter.providers.onnx import ONNXProvider


@pytest.fixture(autouse=True)
def patch_onnx_deps():
    # Ensure the provider uses our mocks
    with patch("aurarouter.providers.onnx.np", mock_np), \
         patch("aurarouter.providers.onnx.ort", mock_ort), \
         patch("aurarouter.providers.onnx._onnx_available", True):
        yield


@pytest.fixture
def mock_session():
    session = MagicMock()
    mock_ort.InferenceSession.return_value = session
    
    # Mock session outputs
    mock_output = MagicMock()
    mock_output.name = "logits"
    session.get_outputs.return_value = [mock_output]
    
    # Default mock run result
    logits = [[0.1, 0.8, 0.1]]
    session.run.return_value = [mock_np.array(logits)]
    
    yield session


@pytest.fixture
def mock_tokenizer_class():
    mock_tok_instance = MagicMock()
    mock_tokenizers.Tokenizer.from_file.return_value = mock_tok_instance
    
    # Mock encoding result
    mock_encoded = MagicMock()
    mock_encoded.ids = [101, 102, 103]
    mock_encoded.attention_mask = [1, 1, 1]
    mock_tok_instance.encode.return_value = mock_encoded
    
    yield mock_tokenizers.Tokenizer


def test_onnx_provider_classification(mock_session, mock_tokenizer_class):
    cfg = {
        "provider": "onnx",
        "model_path": "/fake/model.onnx",
        "tokenizer_path": "/fake/tokenizer.json",
        "task": "classification",
        "labels": ["FAIL", "SUCCESS", "PENDING"],
    }
    # Reset mock_session for this test
    logits = [[0.1, 0.8, 0.1]]
    mock_session.run.return_value = [mock_np.array(logits)]
    
    provider = ONNXProvider(cfg)
    result = provider.generate("test prompt")
    
    assert result == "SUCCESS"


def test_onnx_provider_embedding(mock_session, mock_tokenizer_class):
    # Update mock to return embedding-like vector [batch=1, dim=3]
    vec = [[1.0, 2.0, 3.0]]
    mock_session.run.return_value = [mock_np.array(vec)]
    
    cfg = {
        "provider": "onnx",
        "model_path": "/fake/model.onnx",
        "tokenizer_path": "/fake/tokenizer.json",
        "task": "embedding"
    }
    provider = ONNXProvider(cfg)
    
    result = provider.generate("test prompt")
    
    import json
    data = json.loads(result)
    assert data == [1.0, 2.0, 3.0]


def test_onnx_provider_missing_config():
    provider = ONNXProvider({"provider": "onnx"})
    with pytest.raises(ValueError, match="requires model_path"):
        provider.generate("test")


def test_onnx_provider_no_tokenizer_library(mock_session):
    # Mock ImportError for tokenizers
    with patch("aurarouter.providers.onnx.Tokenizer", side_effect=ImportError, create=True):
        cfg = {
            "provider": "onnx",
            "model_path": "/fake/model.onnx",
            "tokenizer_path": "/fake/tokenizer.json",
        }
        provider = ONNXProvider(cfg)
        # Force re-load attempt
        provider._loaded_tokenizer = False
        with pytest.raises(ValueError, match="requires 'tokenizers' library"):
            provider.generate("test")
