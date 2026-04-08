"""Edge Complexity Scorer — fast, local, zero-dependency complexity estimator.

Stage 1 pre-filter: always runs unconditionally before any intent classifier.
Contributes complexity_score to shared pipeline state.  Never triggers the
pipeline short-circuit — it is an interceptor, not a classifier.

No ML models, no network calls, no external dependencies.
Scoring completes in <1ms for any input.

TG3 — Pluggable Analyzer Pipeline Phase 6
"""

from __future__ import annotations

import re
import time

from aurarouter.analyzer_protocol import AnalysisResult

# ---------------------------------------------------------------------------
# Module-level compiled patterns (compiled once at import time)
# ---------------------------------------------------------------------------

_MULTI_STEP_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bfirst\b", re.IGNORECASE),
    re.compile(r"\bthen\b", re.IGNORECASE),
    re.compile(r"\bfinally\b", re.IGNORECASE),
    re.compile(r"\bnext\b", re.IGNORECASE),
    re.compile(r"\bafterward[s]?\b", re.IGNORECASE),
    re.compile(r"\bstep\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*\d+\.", re.MULTILINE),        # numbered list items
    re.compile(r"^\s*[-*]\s+", re.MULTILINE),       # bullet list items (at least 2)
]

_TECHNICAL_VOCAB: frozenset = frozenset({
    # Architectural / systems terms
    "api", "apis", "microservice", "microservices", "architecture", "distributed",
    "concurrent", "concurrency", "async", "asynchronous", "synchronous",
    "thread", "threads", "mutex", "semaphore", "deadlock", "race", "condition",
    "latency", "throughput", "scalability", "resilience", "fault", "tolerance",
    "kubernetes", "docker", "container", "orchestration", "pipeline",
    "optimize", "optimization", "performance", "benchmark", "profiling",
    "refactor", "refactoring", "abstraction", "interface", "protocol", "schema",
    "database", "sql", "nosql", "query", "index", "transaction", "migration",
    "encryption", "authentication", "authorization", "security", "vulnerability",
    "serialize", "serialization", "deserialization", "codec", "payload",
    "algorithm", "complexity", "heuristic", "recursive", "iteration",
    "neural", "network", "model", "embedding", "inference", "training",
    "dependency", "injection", "inversion", "coupling", "cohesion",
    "monolith", "service", "mesh", "gateway", "proxy", "load", "balancer",
    "event", "sourcing", "saga", "cqrs", "domain", "driven", "design",
    "inheritance", "polymorphism", "encapsulation", "generics", "lambda",
    "decorator", "middleware", "interceptor", "singleton", "factory",
    "observer", "repository", "adapter", "facade", "strategy", "composite",
    "integration", "webhook", "callback", "polling", "streaming",
    "cluster", "shard", "replica", "partition", "consistency", "eventual",
    "cache", "invalidation", "ttl", "eviction", "redis", "memcached",
    "grpc", "graphql", "websocket", "rest", "http", "tcp", "udp",
    # Common programming / code terms (lighter signal — still meaningful)
    "function", "method", "variable", "array", "string", "integer", "boolean",
    "object", "instance", "class", "module", "package", "library", "framework",
    "component", "struct", "enum", "type", "generic", "parameter", "argument",
    "return", "iterator", "generator", "decorator", "annotation", "exception",
    "implement", "implementation", "initialize", "configuration", "config",
    "endpoint", "handler", "middleware", "controller", "service", "provider",
    "interface", "abstract", "inheritance", "override", "polymorphism",
    "sort", "search", "filter", "map", "reduce", "transform", "parse",
    "validate", "sanitize", "encrypt", "hash", "token", "session",
    "request", "response", "header", "body", "payload", "status",
    "error", "exception", "retry", "timeout", "circuit", "breaker",
})

_CONSTRAINT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bmust\s+not\b", re.IGNORECASE),
    re.compile(r"\bwithout\b", re.IGNORECASE),
    re.compile(r"\bexcept\b", re.IGNORECASE),
    re.compile(r"\bunless\b", re.IGNORECASE),
    re.compile(r"\bif\b.{1,60}\bthen\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bwhile\b.{1,60}\bavoid\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bensure\b", re.IGNORECASE),
    re.compile(r"\bcannot\b|\bcan't\b", re.IGNORECASE),
    re.compile(r"\brequire[sd]?\b", re.IGNORECASE),
    re.compile(r"\bconstraint\b", re.IGNORECASE),
    re.compile(r"\bforbid\b|\bforbidden\b", re.IGNORECASE),
]

_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bacross\b", re.IGNORECASE),
    re.compile(r"\bintegrate\b|\bintegration\b", re.IGNORECASE),
    re.compile(r"\bbetween\b.{3,40}\band\b", re.IGNORECASE),
    re.compile(r"\ball\s+of\s+the\b", re.IGNORECASE),
    re.compile(r"\bmultiple\b", re.IGNORECASE),
    re.compile(r"\bsystem[-\s]wide\b", re.IGNORECASE),
    re.compile(r"\bend[-\s]to[-\s]end\b", re.IGNORECASE),
    re.compile(r"\bfull\s+stack\b", re.IGNORECASE),
    re.compile(r"\bmodule[s]?\b", re.IGNORECASE),
    re.compile(r"\bcomponent[s]?\b", re.IGNORECASE),
    re.compile(r"\brepository\b|\brepo\b", re.IGNORECASE),
    re.compile(r"\bservice[s]?\b", re.IGNORECASE),
]

_CODE_INDICATORS: list[re.Pattern] = [
    re.compile(r"```"),                              # code block
    re.compile(r"\bimport\s+\w+", re.IGNORECASE),   # import statement
    re.compile(r"\bclass\s+\w+", re.IGNORECASE),    # class definition
    re.compile(r"\bdef\s+\w+", re.IGNORECASE),      # function definition
    re.compile(r"\bfunction\s+\w+", re.IGNORECASE), # JS function
    re.compile(r"\bvoid\s+\w+\s*\("),               # C/C++/Java method
    re.compile(r"\bpublic\s+\w+\s+\w+\s*\("),       # Java/C# method
    re.compile(r"\breturn\s+\w+"),                   # return statement
    re.compile(r"\bif\s*\(.+\)\s*\{"),              # if block (code style)
    re.compile(r"\bfor\s*\(.+\)\s*\{"),             # for loop (code style)
    re.compile(r"::\w+"),                            # Rust/C++ scope resolution
    re.compile(r"=>\s*\{"),                          # arrow function body
]

_WEIGHTS = {
    "length": 0.15,
    "structure": 0.20,
    "vocabulary": 0.20,
    "constraints": 0.15,
    "scope": 0.15,
    "code": 0.15,
}


class EdgeComplexityScorer:
    """Fast, local mathematical complexity estimator.

    Stage 1 pre-filter: always runs unconditionally before any intent
    classifier.  Contributes complexity_score to shared pipeline state.
    Never triggers pipeline short-circuit.

    No ML models, no network calls, no external dependencies.
    """

    analyzer_id: str = "edge-complexity"
    priority: int = 200  # Highest priority in Stage 1 pre-filter registry

    def __init__(
        self,
        simple_ceiling: int = 3,
        complex_floor: int = 7,
    ) -> None:
        """
        Args:
            simple_ceiling: Complexity at or below this = simple task.
            complex_floor: Complexity at or above this = complex task.
        """
        self._simple_ceiling = simple_ceiling
        self._complex_floor = complex_floor

    # ── PromptAnalyzer protocol ──────────────────────────────────────

    @property
    def analyzer_id(self) -> str:  # type: ignore[override]
        return "edge-complexity"

    @property
    def priority(self) -> int:  # type: ignore[override]
        return 200

    def supports(self, prompt: str) -> bool:
        """Always returns True — complexity is always measurable."""
        return True

    def analyze(self, prompt: str, context: str = "") -> AnalysisResult:
        """Compute complexity score from feature vector.

        Returns an AnalysisResult with:
          - complexity_score: 1–10 (always non-None, never abstains)
          - confidence: 1.0 (the measurement is always definitive)
          - intent: hint-only (Stage 2 classifiers will override)
        """
        score = self._compute_score(prompt)

        # Intent inference is a hint only — Stage 2 classifiers override this
        if score <= self._simple_ceiling:
            hint_intent = "DIRECT" if score <= 1 else "SIMPLE_CODE"
        elif score >= self._complex_floor:
            hint_intent = "COMPLEX_REASONING"
        else:
            hint_intent = "SIMPLE_CODE"  # Conservative default

        return AnalysisResult(
            intent=hint_intent,
            confidence=1.0,               # Measurement is always definitive
            complexity_score=score,
            analyzer_id="edge-complexity",
            reasoning=f"Complexity={score} (length+structure+vocab+constraints+scope+code)",
            metadata={
                "simple_ceiling": self._simple_ceiling,
                "complex_floor": self._complex_floor,
            },
        )

    # ── Feature extraction ───────────────────────────────────────────

    def _compute_score(self, prompt: str) -> int:
        """Compute weighted complexity score, clamped to [1, 10]."""
        if not prompt:
            return 1

        raw = (
            self._score_length(prompt) * _WEIGHTS["length"]
            + self._score_structure(prompt) * _WEIGHTS["structure"]
            + self._score_vocabulary(prompt) * _WEIGHTS["vocabulary"]
            + self._score_constraints(prompt) * _WEIGHTS["constraints"]
            + self._score_scope(prompt) * _WEIGHTS["scope"]
            + self._score_code(prompt) * _WEIGHTS["code"]
        )
        return max(1, min(10, round(raw * 10)))

    def _score_length(self, prompt: str) -> float:
        """Normalized length.  <50 chars = 0.0, >2000 chars = 1.0, linear."""
        n = len(prompt)
        if n <= 50:
            return 0.0
        if n >= 2000:
            return 1.0
        return (n - 50) / (2000 - 50)

    def _score_structure(self, prompt: str) -> float:
        """Multi-step and structural indicators (0.0–1.0)."""
        hits = 0
        for pat in _MULTI_STEP_PATTERNS:
            if pat.search(prompt):
                hits += 1
        # Cap: 3+ structural signals = 1.0
        return min(1.0, hits / 3.0)

    def _score_vocabulary(self, prompt: str) -> float:
        """Technical vocabulary density (unique tech terms / total unique words)."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", prompt.lower())
        if not words:
            return 0.0
        unique_words = set(words)
        tech_hits = len(unique_words & _TECHNICAL_VOCAB)
        # Density: tech_hits / total unique words, capped at 1.0
        density = tech_hits / max(1, len(unique_words))
        # Scale: 20%+ density (of unique words) = 1.0
        return min(1.0, density / 0.20)

    def _score_constraints(self, prompt: str) -> float:
        """Negation and conditional constraint density (0.0–1.0)."""
        hits = 0
        for pat in _CONSTRAINT_PATTERNS:
            if pat.search(prompt):
                hits += 1
        # Cap: 4+ constraint signals = 1.0
        return min(1.0, hits / 4.0)

    def _score_scope(self, prompt: str) -> float:
        """Cross-component and breadth indicators (0.0–1.0)."""
        hits = 0
        for pat in _SCOPE_PATTERNS:
            if pat.search(prompt):
                hits += 1
        # Cap: 3+ scope signals = 1.0
        return min(1.0, hits / 3.0)

    def _score_code(self, prompt: str) -> float:
        """Code block and code structure presence (0.0–1.0)."""
        hits = 0
        for pat in _CODE_INDICATORS:
            if pat.search(prompt):
                hits += 1
        # Cap: 3+ code signals = 1.0
        return min(1.0, hits / 3.0)
