"""General ONNX inference provider.

Loads any ONNX model and tokenizer for local execution.
Supports classification, embedding, and basic sequence-to-sequence tasks.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    import onnxruntime as ort
    _onnx_available = True
except ImportError:
    np = None  # type: ignore
    ort = None  # type: ignore
    _onnx_available = False

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.ONNXProvider")


class ONNXSessionCache:
    """Thread-safe cache for ONNX sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_session(self, model_path: str, providers: list[str] | None = None) -> Any:
        if not _onnx_available:
            raise ImportError("onnxruntime is not installed. Install with: pip install aurarouter[vector]")
        
        with self._lock:
            if model_path not in self._sessions:
                # Default to CPU if no providers specified
                prov = providers or ["CPUExecutionProvider"]
                logger.info("Loading ONNX model: %s (providers=%s)", model_path, prov)
                self._sessions[model_path] = ort.InferenceSession(model_path, providers=prov)
            return self._sessions[model_path]


_session_cache = ONNXSessionCache()


class ONNXProvider(BaseProvider):
    """General ONNX inference provider.

    Configuration in auraconfig.yaml:
        model_id:
            provider: onnx
            model_path: /path/to/model.onnx
            tokenizer_path: /path/to/tokenizer.json  # Optional
            task: classification | generation | embedding
            labels: ["LABEL_0", "LABEL_1", ...]  # For classification
            parameters:
                max_length: 128
                input_names: ["input_ids", "attention_mask"]
                output_name: "logits"
                providers: ["CPUExecutionProvider"]
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self._tokenizer = None
        self._loaded_tokenizer = False

    def _get_tokenizer(self):
        """Lazy load tokenizer using tokenizers library if available."""
        if self._loaded_tokenizer:
            return self._tokenizer
        
        self._loaded_tokenizer = True
        tokenizer_path = self.config.get("tokenizer_path")
        if not tokenizer_path:
            return None

        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.debug("Loaded tokenizer from %s", tokenizer_path)
        except ImportError:
            logger.warning("tokenizers library not installed. Falling back to simple whitespace tokenization.")
        except Exception as e:
            logger.warning("Failed to load tokenizer from %s: %s", tokenizer_path, e)
        
        return self._tokenizer

    def generate(self, prompt: str, json_mode: bool = False,
                 response_schema: dict | None = None) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode,
                                        response_schema=response_schema).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False,
        response_schema: dict | None = None,
    ) -> GenerateResult:
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("ONNXProvider requires model_path in configuration.")

        params = self.config.get("parameters", {})
        providers = params.get("providers", ["CPUExecutionProvider"])
        
        try:
            session = _session_cache.get_session(model_path, providers=providers)
        except Exception as e:
            logger.error("Failed to load ONNX session for %s: %s", model_path, e)
            return GenerateResult(text=f"ERROR: Failed to load ONNX model: {e}")

        task = self.config.get("task", "classification")
        
        # Tokenize
        tokenizer = self._get_tokenizer()
        max_length = params.get("max_length", 128)
        
        if tokenizer:
            encoded = tokenizer.encode(prompt)
            # Truncate if needed
            input_ids = encoded.ids[:max_length]
            attention_mask = encoded.attention_mask[:max_length]
        else:
            # Fallback simple tokenization (similar to ONNXVectorAnalyzer)
            tokens = prompt.lower().split()[:max_length]
            # This is very limited as we don't have the vocab here easily 
            # without the tokenizers library.
            # If tokenizer_path was provided but loading failed, we are in trouble.
            # For now, let's assume if they have a tokenizer_path, they should have the library.
            raise ValueError("ONNXProvider requires 'tokenizers' library to use tokenizer_path.")

        # Prepare inputs
        input_names = params.get("input_names", ["input_ids", "attention_mask"])
        inputs = {}
        if "input_ids" in input_names:
            inputs["input_ids"] = np.array([input_ids], dtype=np.int64)
        if "attention_mask" in input_names:
            inputs["attention_mask"] = np.array([attention_mask], dtype=np.int64)
        
        # Run inference
        try:
            outputs = session.run(None, inputs)
        except Exception as e:
            logger.error("ONNX inference failed: %s", e)
            return GenerateResult(text=f"ERROR: ONNX inference failed: {e}")

        # Process output
        output_idx = 0 # Default to first output
        output_name = params.get("output_name")
        if output_name:
            for i, out in enumerate(session.get_outputs()):
                if out.name == output_name:
                    output_idx = i
                    break
        
        data = outputs[output_idx]

        if task == "classification":
            # Assume data is logits [batch, num_classes]
            logits = np.array(data[0])
            probs = self._softmax(logits)
            idx = np.argmax(probs)
            labels = self.config.get("labels", [])
            if labels and idx < len(labels):
                result_text = labels[idx]
            else:
                result_text = str(idx)
            
            return GenerateResult(
                text=result_text,
                metadata={"confidence": float(probs[idx]), "logits": logits.tolist()}
            )
        
        elif task == "embedding":
            # Return JSON encoded vector
            # Assume data is [batch, hidden_dim] or [batch, seq, hidden_dim]
            if len(data.shape) == 3:
                # Mean pooling
                vec = np.mean(data[0], axis=0)
            else:
                vec = data[0]
            
            vec = np.array(vec)
            return GenerateResult(
                text=json.dumps(vec.tolist()),
                metadata={"embedding_dim": len(vec)}
            )
        
        elif task == "generation":
            # Simple one-shot decoding (if the model returns token IDs)
            # This is NOT a full autoregressive loop yet.
            # Many T5 or small models might return token IDs in one shot if exported that way.
            if data.dtype in (np.int64, np.int32):
                token_ids = data[0].tolist()
                if tokenizer:
                    result_text = tokenizer.decode(token_ids)
                else:
                    result_text = str(token_ids)
                return GenerateResult(text=result_text)
            else:
                return GenerateResult(text="ERROR: generation task requires integer token ID output from model.")

        return GenerateResult(text=str(data.tolist()))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
