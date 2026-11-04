"""SimulatedProvider: Deterministic, software-defined provider for combinatorial testing.

Enables zero-network testing of multi-factor routing conditions:
- RateLimitExceededError (429) injection and RateLimitSnapshot headers
- ContextLengthExceededError (400) context cliff enforcement
- ProviderTransientError (503) circuit breaker trip and recovery
- Injectable virtual clock and TTFT latency simulation
- Streaming SSE emulation with pre-token deadlines
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable

from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.providers.base import BaseProvider
from aurarouter.ratelimit.models import RateLimitSnapshot
from aurarouter.savings.models import GenerateResult


class SimulatedProvider(BaseProvider):
    """Software-defined provider with deterministic fault schedules, rate limits, and context cliffs."""

    def __init__(
        self,
        name: str = "simulated",
        model_id: str = "sim-model-1",
        *,
        context_limit: int = 128_000,
        ttft_s: float = 0.02,
        latency_s: float = 0.05,
        default_output_tokens: int = 64,
        fault_schedule: list[str | Exception] | None = None,
        rate_limit_snapshot: RateLimitSnapshot | None = None,
        clock_fn: Callable[[], float] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_config or {})
        self.name = name
        self.model_id = model_id
        self.context_limit = context_limit
        self.ttft_s = ttft_s
        self.latency_s = latency_s
        self.default_output_tokens = default_output_tokens
        self.fault_schedule = list(fault_schedule) if fault_schedule else []
        self.rate_limit_snapshot = rate_limit_snapshot
        self.clock_fn = clock_fn or time.time

        self.call_count: int = 0
        self.prompts_received: list[str] = []
        self.histories_received: list[list[dict[str, Any]]] = []

    def _next_fault(self) -> None:
        if not self.fault_schedule:
            return
        fault = self.fault_schedule.pop(0)
        if isinstance(fault, Exception):
            raise fault
        if fault in ("429", "rate_limit"):
            snap = self.rate_limit_snapshot or RateLimitSnapshot(
                requests_remaining=0,
                retry_after_s=5.0,
            )
            raise RateLimitExceededError(
                f"Rate limit exceeded on {self.name}",
                snapshot=snap,
                retry_after_s=snap.retry_after_s,
            )
        if fault in ("400", "context_length"):
            raise ContextLengthExceededError(
                f"Context length exceeded on {self.name} (limit={self.context_limit})",
                max_tokens=self.context_limit,
            )
        if fault in ("500", "503", "transient"):
            raise ProviderTransientError(f"Simulated transient error 503 on {self.name}")
        if fault not in ("200", "ok"):
            raise RuntimeError(f"Unknown fault trigger: {fault}")

    def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        response_schema: dict | None = None,
        return_tokens: bool = False,
    ) -> str:
        res = self.generate_with_usage(
            prompt,
            json_mode=json_mode,
            response_schema=response_schema,
            return_tokens=return_tokens,
        )
        return res.text

    def generate_with_usage(self, prompt: str, **kwargs: Any) -> GenerateResult:
        self.call_count += 1
        self.prompts_received.append(prompt)
        self._next_fault()

        est_in = max(1, len(prompt) // 4)
        max_out = int(kwargs.get("max_tokens", self.default_output_tokens))
        if est_in + max_out > self.context_limit:
            raise ContextLengthExceededError(
                f"Prompt ({est_in} tokens) + max_tokens ({max_out}) exceeds context limit ({self.context_limit})",
                max_tokens=self.context_limit,
            )

        finish_reason = kwargs.get("force_finish_reason", "stop")
        return GenerateResult(
            text=f"[{self.name}] response to: {prompt[:30]}",
            input_tokens=est_in,
            output_tokens=max_out,
            model_id=self.model_id,
            provider=self.name,
            finish_reason=finish_reason,
            rate_limit=self.rate_limit_snapshot,
            ttft_s=self.ttft_s,
            metadata={"simulated": True, "call_index": self.call_count},
        )

    def generate_with_history(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self.call_count += 1
        self.histories_received.append(messages)
        self._next_fault()

        total_text = " ".join(str(m.get("content", "")) for m in messages)
        if system_prompt:
            total_text = f"{system_prompt} {total_text}"
        est_in = max(1, len(total_text) // 4)
        max_out = int(kwargs.get("max_tokens", self.default_output_tokens))

        if est_in + max_out > self.context_limit:
            raise ContextLengthExceededError(
                f"Session tokens ({est_in}) + max_tokens ({max_out}) exceeds context limit ({self.context_limit})",
                max_tokens=self.context_limit,
            )

        finish_reason = kwargs.get("force_finish_reason", "stop")
        return GenerateResult(
            text=f"[{self.name}] session reply ({len(messages)} msgs)",
            input_tokens=est_in,
            output_tokens=max_out,
            model_id=self.model_id,
            provider=self.name,
            finish_reason=finish_reason,
            rate_limit=self.rate_limit_snapshot,
            ttft_s=self.ttft_s,
            metadata={"simulated": True, "call_index": self.call_count},
        )

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        self.call_count += 1
        self.prompts_received.append(prompt)
        self._next_fault()

        est_in = max(1, len(prompt) // 4)
        max_out = int(kwargs.get("max_tokens", self.default_output_tokens))
        if est_in + max_out > self.context_limit:
            raise ContextLengthExceededError(
                f"Stream prompt ({est_in}) exceeds context limit ({self.context_limit})",
                max_tokens=self.context_limit,
            )

        res = GenerateResult(
            text=f"[{self.name}] streamed response",
            input_tokens=est_in,
            output_tokens=max_out,
            model_id=self.model_id,
            provider=self.name,
            finish_reason="stop",
            rate_limit=self.rate_limit_snapshot,
            ttft_s=self.ttft_s,
            metadata={"simulated": True},
        )
        self._record_stream_result(res)

        tokens = [f"[{self.name}] ", "streamed ", "token ", "chunk."]
        for tok in tokens:
            yield tok

    def health_check(self) -> bool:
        if self.fault_schedule and self.fault_schedule[0] in ("500", "503", "transient"):
            return False
        return True
