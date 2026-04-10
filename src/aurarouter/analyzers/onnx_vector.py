"""ONNX-based sentence embedding classifier for AuraRouter.

Stage 2 classifier: runs in priority order, short-circuits when confidence
meets the threshold.  Embeds the prompt using a lightweight sentence
transformer (all-MiniLM-L6-v2 quantized) distributed via the
aurarouter-onnx companion sidecar package.

Degrades gracefully when aurarouter-onnx or onnxruntime is not installed.

TG2 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aurarouter.analyzer_protocol import AnalysisResult

if TYPE_CHECKING:
    from aurarouter.intent_registry import IntentRegistry

logger = logging.getLogger("AuraRouter.ONNXVectorAnalyzer")

_WARNED_MISSING: set[str] = set()


class ONNXVectorAnalyzer:
    """ONNX-based sentence embedding classifier.

    Stage 2 classifier (intent classification with confidence-based short-circuit).
    Embeds the prompt using a lightweight sentence transformer distributed via the
    aurarouter-onnx companion package, then compares against pre-computed intent
    embeddings via cosine similarity.

    High priority in Stage 2 (runs first among intent classifiers).
    """

    # Class-level constants (accessed both as class and instance attributes)
    _ANALYZER_ID = "onnx-vector"
    _PRIORITY = 100  # Highest priority in Stage 2 classifier registry

    def __init__(
        self,
        intent_registry: IntentRegistry,
        similarity_threshold: float = 0.70,
        margin_threshold: float = 0.10,
        max_sequence_length: int = 128,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        """
        Args:
            intent_registry: Registry of intent definitions for embedding matrix.
            similarity_threshold: Minimum cosine similarity to report a match.
            margin_threshold: Minimum gap between top-1 and top-2 similarity
                              for high confidence.  Below this, confidence is reduced.
            max_sequence_length: Truncation length for tokenization.
            model_path: Explicit override path.  If None, resolves from
                        internal package data.
            tokenizer_path: Explicit override path for tokenizer.json.
        """
        self._registry = intent_registry
        self._similarity_threshold = similarity_threshold
        self._margin_threshold = margin_threshold
        self._max_sequence_length = max_sequence_length

        # Resolve paths from internal package data or explicit overrides
        self._model_path: str | None = model_path or self._resolve_companion_model_path()
        self._tokenizer_path: str | None = tokenizer_path or self._resolve_companion_tokenizer_path()

        # Session and matrices are initialized lazily on first analyze() call
        self._session = None          # onnxruntime.InferenceSession or None
        self._intent_matrix = None    # np.ndarray or None
        self._intent_names: list[str] = []
        self._vocab: dict[str, int] = {}
        self._loaded: bool = False

    # ── PromptAnalyzer protocol ──────────────────────────────────────

    @property
    def analyzer_id(self) -> str:
        return self._ANALYZER_ID

    @property
    def priority(self) -> int:
        return self._PRIORITY

    def supports(self, prompt: str) -> bool:
        """Returns True if aurarouter-onnx package is installed and model available."""
        if not self._loaded:
            self._try_load()
        return self._session is not None

    def analyze(self, prompt: str, context: str = "") -> AnalysisResult | None:
        """Classify prompt using ONNX cosine similarity.

        Returns None if:
          - aurarouter-onnx package not installed
          - onnxruntime not installed
          - model file missing
          - max_similarity below similarity_threshold

        complexity_score is always 0 (sentinel) — Stage 1 pre-filter owns it.
        """
        if not self.supports(prompt):
            return None

        try:
            import numpy as np

            query_vec = self._embed(prompt)
            similarities = self._cosine_similarity_batch(query_vec)

            if similarities is None or len(similarities) == 0:
                return None

            # Top-1 and Top-2 similarities
            sorted_idx = np.argsort(similarities)[::-1]
            top1_idx = sorted_idx[0]
            top1_sim = float(similarities[top1_idx])

            if top1_sim < self._similarity_threshold:
                return None  # Abstain — low similarity

            # Confidence from similarity and margin
            if len(sorted_idx) >= 2:
                top2_sim = float(similarities[sorted_idx[1]])
                margin = top1_sim - top2_sim
            else:
                margin = self._margin_threshold  # Single intent — treat as full margin

            if margin >= self._margin_threshold:
                confidence = top1_sim
            else:
                # Reduce confidence proportionally to margin deficit
                confidence = top1_sim * (margin / self._margin_threshold)

            intent = self._intent_names[top1_idx]

            return AnalysisResult(
                intent=intent,
                confidence=confidence,
                complexity_score=0,  # Sentinel: Stage 1 pre-filter owns complexity
                analyzer_id=self._ANALYZER_ID,
                reasoning=f"ONNX cosine similarity: {top1_sim:.3f} margin: {margin:.3f}",
                metadata={
                    "top1_similarity": top1_sim,
                    "margin": margin,
                    "embedding_dim": len(query_vec),
                },
            )
        except Exception as exc:
            logger.warning("ONNXVectorAnalyzer.analyze failed: %s", exc, exc_info=True)
            return None

    # ── Companion package resolution ─────────────────────────────────

    @staticmethod
    def _resolve_companion_model_path() -> str | None:
        """Locate the ONNX model file bundled as internal package data."""
        try:
            import importlib.resources as resources
            # Get the path to the internal resource file
            # Traverses into aurarouter.resources.onnx
            ref = resources.files("aurarouter.resources.onnx") / "sentence_encoder.onnx"
            if ref.exists():
                return str(ref)
            # Try fallback to absolute path relative to this file
            import os
            base = os.path.dirname(os.path.dirname(__file__))
            fallback = os.path.join(base, "resources", "onnx", "sentence_encoder.onnx")
            if os.path.exists(fallback):
                return fallback
            return None
        except (ImportError, AttributeError):
            return None

    @staticmethod
    def _resolve_companion_tokenizer_path() -> str | None:
        """Locate tokenizer.json from internal package data."""
        try:
            import importlib.resources as resources
            ref = resources.files("aurarouter.resources.onnx") / "tokenizer.json"
            if ref.exists():
                return str(ref)
            # Try fallback
            import os
            base = os.path.dirname(os.path.dirname(__file__))
            fallback = os.path.join(base, "resources", "onnx", "tokenizer.json")
            if os.path.exists(fallback):
                return fallback
            return None
        except (ImportError, AttributeError):
            return None

    # ── Lazy initialization ──────────────────────────────────────────

    def _try_load(self) -> None:
        """Attempt to initialize ONNX session.  Safe to call multiple times."""
        if self._loaded:
            return
        self._loaded = True  # Mark loaded regardless — prevents repeated attempts

        # Check onnxruntime availability
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError:
            if "onnxruntime" not in _WARNED_MISSING:
                logger.warning(
                    "onnxruntime not installed — ONNXVectorAnalyzer disabled. "
                    "Install with: pip install aurarouter[vector]"
                )
                _WARNED_MISSING.add("onnxruntime")
            return

        # Check model path
        if not self._model_path:
            logger.warning(
                "ONNX model file not found in package data — ONNXVectorAnalyzer disabled"
            )
            return

        import os
        if not os.path.exists(self._model_path):
            logger.warning(
                "ONNX model file not found: %s — ONNXVectorAnalyzer disabled",
                self._model_path,
            )
            return

        try:
            self._session = ort.InferenceSession(
                self._model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNXVectorAnalyzer: loaded model from %s", self._model_path)
            # Load tokenizer vocabulary
            self._load_tokenizer()
            # Pre-compute intent embeddings
            self._build_intent_matrix()
        except Exception as exc:
            logger.warning("Failed to load ONNX model: %s", exc, exc_info=True)
            self._session = None

    def _load_tokenizer(self) -> None:
        """Load vocabulary from tokenizer.json (companion package)."""
        if not self._tokenizer_path:
            return
        import json, os
        if not os.path.exists(self._tokenizer_path):
            return
        try:
            with open(self._tokenizer_path, "r", encoding="utf-8") as f:
                tok_data = json.load(f)
            # Standard HuggingFace tokenizer format
            model_data = tok_data.get("model", tok_data)
            self._vocab = model_data.get("vocab", {})
        except Exception as exc:
            logger.warning("Failed to load tokenizer: %s", exc)

    def _build_intent_matrix(self) -> None:
        """Pre-compute embeddings for all registered intents."""
        import numpy as np

        intents = self._registry.get_all()
        if not intents:
            return
        texts = [f"{i.name}: {i.description}" for i in intents]
        embeddings = []
        for t in texts:
            vec = self._embed(t)
            if vec is not None:
                embeddings.append(vec)

        if not embeddings:
            return

        matrix = np.stack(embeddings, axis=0).astype(np.float32)
        # L2 normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._intent_matrix = matrix / norms
        self._intent_names = [i.name for i in intents]

    # ── Embedding utilities ──────────────────────────────────────────

    def _embed(self, text: str):
        """Embed a single text string.  Returns float32 L2-normalized vector."""
        if self._session is None:
            return None
        try:
            import numpy as np

            input_ids, attention_mask = self._tokenize(text)
            # Run ONNX inference
            # Standard sentence-transformer input names
            try:
                outputs = self._session.run(
                    None,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                )
            except Exception:
                # Try alternate input names (some ONNX exports differ)
                outputs = self._session.run(None, {"input_ids": input_ids})

            # last_hidden_state: [batch, seq, hidden_dim]
            last_hidden = outputs[0].astype(np.float32)
            mask_expanded = attention_mask[..., None].astype(np.float32)
            # Mean pooling over non-padding tokens
            summed = (last_hidden * mask_expanded).sum(axis=1)
            count = mask_expanded.sum(axis=1)
            mean_pooled = summed / np.maximum(count, 1e-9)
            vec = mean_pooled[0]  # [hidden_dim]
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32)
        except Exception as exc:
            logger.debug("_embed failed: %s", exc)
            return None

    def _tokenize(self, text: str):
        """Tokenize text to input_ids and attention_mask numpy arrays."""
        import numpy as np

        # Simple whitespace tokenization with vocabulary lookup
        # Falls back to character-level IDs if no vocabulary loaded
        tokens = text.lower().split()[:self._max_sequence_length]

        if self._vocab:
            # Map tokens → vocab IDs; unknown → UNK (0 or special token)
            unk_id = self._vocab.get("[UNK]", 0)
            cls_id = self._vocab.get("[CLS]", 101)
            sep_id = self._vocab.get("[SEP]", 102)
            ids = [cls_id] + [self._vocab.get(t, unk_id) for t in tokens] + [sep_id]
        else:
            # Fallback: use character ordinals (functional but not semantic)
            ids = [ord(c) % 30522 for c in text[:self._max_sequence_length]]

        seq_len = len(ids)
        input_ids = np.array([ids], dtype=np.int64)                       # [1, seq]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)            # [1, seq]
        return input_ids, attention_mask

    def _cosine_similarity_batch(self, query) -> "np.ndarray | None":
        """Vectorized cosine similarity: query vs all intents.

        Both query and intent_matrix are L2-normalized, so dot product = cosine sim.
        """
        if self._intent_matrix is None or query is None:
            return None
        import numpy as np
        q = query.astype(np.float32)
        return (q @ self._intent_matrix.T).astype(np.float32)
