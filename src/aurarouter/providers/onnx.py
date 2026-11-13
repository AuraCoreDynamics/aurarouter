"""General ONNX inference provider.

Loads any ONNX model and tokenizer for local execution.
Supports classification, embedding, and basic sequence-to-sequence tasks.
"""

from __future__ import annotations

import json
import threading
from typing import Any

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
            raise ImportError("onnxruntime is not installed. Install with: pip install aurarouter")
        
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

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
        images: list[dict] | None = None,
    ) -> str:
        return self.generate_with_usage(
            prompt,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
            images=images,
        ).text

    def _preprocess_images(self, images: list[dict], params: dict) -> Any:
        """Decode and normalize image payloads into an NCHW float32 numpy tensor."""
        if np is None:
            raise ImportError("numpy is required for ONNX image preprocessing.")
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for ONNXProvider image preprocessing. "
                "Install with: pip install Pillow"
            ) from exc

        import base64
        import io
        from aurarouter.media import normalize_images

        norm_images = normalize_images(images)
        if not norm_images:
            raise ValueError("No valid images provided for ONNX vision inference.")

        raw_size = params.get("image_size", 224)
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            target_h, target_w = int(raw_size[0]), int(raw_size[1])
        else:
            target_h = target_w = int(raw_size)

        mean = np.array(params.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(params.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 3, 1, 1)

        batch: list[Any] = []
        for img_input in norm_images:
            raw_bytes = base64.b64decode(img_input.data)
            with Image.open(io.BytesIO(raw_bytes)) as im:
                rgb = im.convert("RGB").resize((target_w, target_h))
                arr = np.asarray(rgb, dtype=np.float32) / 255.0  # HWC
                chw = np.transpose(arr, (2, 0, 1))  # CHW
                batch.append(chw)

        tensor = np.stack(batch, axis=0)
        return (tensor - mean) / std

    def generate_with_usage(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
        images: list[dict] | None = None,
    ) -> GenerateResult:
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("ONNXProvider requires model_path in configuration.")

        if images and not self.supports_modality("vision"):
            raise ValueError(
                "ONNXProvider model does not declare 'vision' modality support "
                "(set modalities/capabilities/tags to include 'vision' or configure a vision task)."
            )

        params = self.config.get("parameters", {})
        providers = params.get("providers", ["CPUExecutionProvider"])

        try:
            session = _session_cache.get_session(model_path, providers=providers)
        except Exception as e:
            logger.error("Failed to load ONNX session for %s: %s", model_path, e)
            return GenerateResult(text=f"ERROR: Failed to load ONNX model: {e}")

        task = str(self.config.get("task", "classification")).strip().lower()
        image_input_name = str(params.get("image_input_name", "pixel_values"))
        default_inputs = [image_input_name] if images and not self.config.get("tokenizer_path") else ["input_ids", "attention_mask"]
        if images and self.config.get("tokenizer_path") and "input_names" not in params:
            default_inputs = ["input_ids", "attention_mask", image_input_name]
        input_names = list(params.get("input_names", default_inputs))
        inputs: dict[str, Any] = {}

        needs_text = any(name in input_names for name in ("input_ids", "attention_mask"))
        tokenizer = self._get_tokenizer() if needs_text else None
        max_length = params.get("max_length", 128)

        if needs_text:
            if tokenizer:
                encoded = tokenizer.encode(prompt)
                input_ids = encoded.ids[:max_length]
                attention_mask = encoded.attention_mask[:max_length]
            else:
                # Without the tokenizers library, ONNXProvider cannot map tokens to vocab IDs.
                raise ValueError("ONNXProvider requires 'tokenizers' library to use tokenizer_path.")

            if "input_ids" in input_names:
                inputs["input_ids"] = np.array([input_ids], dtype=np.int64)
            if "attention_mask" in input_names:
                inputs["attention_mask"] = np.array([attention_mask], dtype=np.int64)

        if images:
            inputs[image_input_name] = self._preprocess_images(images, params)

        # Run inference
        try:
            outputs = session.run(None, inputs)
        except Exception as e:
            logger.error("ONNX inference failed: %s", e)
            return GenerateResult(text=f"ERROR: ONNX inference failed: {e}")

        # Process output
        output_idx = 0  # Default to first output
        output_name = params.get("output_name")
        if output_name:
            for i, out in enumerate(session.get_outputs()):
                if out.name == output_name:
                    output_idx = i
                    break

        data = outputs[output_idx]

        if task in ("classification", "vision_classification", "image_classification"):
            # Assume data is logits [batch, num_classes]
            logits = np.array(data[0])
            probs = self._softmax(logits)
            idx = int(np.argmax(probs))
            labels = self.config.get("labels", [])
            if labels and idx < len(labels):
                result_text = labels[idx]
            else:
                result_text = str(idx)

            return GenerateResult(
                text=result_text,
                metadata={"confidence": float(probs[idx]), "logits": logits.tolist()},
            )

        elif task in ("embedding", "vision_embedding"):
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
                metadata={"embedding_dim": len(vec)},
            )

        elif task in ("generation", "multimodal"):
            # Simple one-shot decoding (if the model returns token IDs)
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
