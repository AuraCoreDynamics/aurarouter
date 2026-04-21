"""Inference auction and telemetry contracts for AuraRouter grid integration.

Phase 0 — Contract Definitions Only. No implementation logic.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelState(str, Enum):
    WARM = "warm"
    COLD = "cold"
    LOADING = "loading"
    UNKNOWN = "unknown"


class ModelTelemetry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(alias="modelId")
    provider_name: str = Field(alias="providerName")
    state: ModelState
    vram_usage_mb: float | None = Field(default=None, alias="vramUsageMb")
    last_used: datetime | None = Field(default=None, alias="lastUsed")
    context_slots_free: int | None = Field(default=None, alias="contextSlotsFree")


class ProviderHealthState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider_name: str = Field(alias="providerName")
    is_healthy: bool = Field(alias="isHealthy")
    consecutive_failures: int = Field(default=0, alias="consecutiveFailures")
    last_success: datetime | None = Field(default=None, alias="lastSuccess")
    last_failure: datetime | None = Field(default=None, alias="lastFailure")
    circuit_state: str = Field(default="closed", alias="circuitState")


class InferenceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    model_id: str = Field(alias="modelId")
    prompt_token_estimate: int = Field(alias="promptTokenEstimate")
    max_tokens: int = Field(alias="maxTokens")
    vram_requirement_mb: float | None = Field(default=None, alias="vramRequirementMb")
    is_transient: bool = Field(default=False, alias="isTransient")
    timeout_ms: int = Field(default=30000, alias="timeoutMs")
    originator_node_id: str = Field(alias="originatorNodeId")


class InferenceBid(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(alias="requestId")
    node_id: str = Field(alias="nodeId")
    score: float
    is_warm: bool = Field(alias="isWarm")
    estimated_latency_ms: int = Field(alias="estimatedLatencyMs")
    vram_free_mb: float | None = Field(default=None, alias="vramFreeMb")
    bid_timestamp: datetime = Field(alias="bidTimestamp")

    @field_validator("score")
    @classmethod
    def score_must_be_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        return v


class ICapacityAdvisor(Protocol):
    def get_preload_hints(self) -> list[str]: ...
    def adjust_bid_score(self, bid: InferenceBid, request: InferenceRequest) -> float: ...
