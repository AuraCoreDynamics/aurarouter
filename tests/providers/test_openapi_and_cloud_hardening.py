"""Tests for OpenAPIProvider completeness and cloud-endpoint auto-classification hardening."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import httpx
import pytest
import respx

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.providers.openapi import OpenAPIProvider
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.models import GenerateResult
from aurarouter.savings.pricing import (
    CostEngine,
    PricingCatalog,
    infer_hosting_tier_from_endpoint,
    is_cloud_tier,
    resolve_hosting_tier,
)
from aurarouter.savings.usage_store import UsageStore


class TestCloudEndpointHardening:
    def test_infer_hosting_tier_from_endpoint(self):
        assert infer_hosting_tier_from_endpoint("https://api.openai.com/v1") == "cloud"
        assert infer_hosting_tier_from_endpoint("https://contoso.openai.azure.com/openai") == "cloud"
        assert infer_hosting_tier_from_endpoint("https://api.groq.com/openai/v1") == "cloud"
        assert infer_hosting_tier_from_endpoint("https://openrouter.ai/api/v1") == "cloud"
        assert infer_hosting_tier_from_endpoint("https://api.anthropic.com/v1") == "cloud"

        # Local / LAN / internal endpoints remain None -> default on-prem
        assert infer_hosting_tier_from_endpoint("http://localhost:8000/v1") is None
        assert infer_hosting_tier_from_endpoint("http://127.0.0.1:11434/v1") is None
        assert infer_hosting_tier_from_endpoint("http://192.168.1.50:8000/v1") is None
        assert infer_hosting_tier_from_endpoint("http://vllm.internal.corp:8000/v1") is None

    def test_is_cloud_tier_and_openai_builtin_prices(self):
        assert PricingCatalog.is_cloud_provider("openai") is True
        assert PricingCatalog.is_cloud_provider("azure_openai") is True
        assert PricingCatalog.is_cloud_provider("openapi") is False  # Unqualified local default preserved

        # Unqualified openapi pointing at api.openai.com auto-upgrades to cloud!
        assert is_cloud_tier(None, "openapi", endpoint="https://api.openai.com/v1") is True
        # Explicit override is still respected if user explicitly sets hosting_tier
        assert resolve_hosting_tier("on-prem", "openapi", endpoint="https://api.openai.com/v1") == "on-prem"

        cat = PricingCatalog()
        assert cat.get_price("gpt-4o", "openai").input_per_million == 2.50
        assert cat.get_price("o3", "openai").output_per_million == 8.00

    def test_budget_and_local_chain_block_disguised_openapi_cloud_endpoint(self, tmp_path):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "routing_stack": {"mode": "active", "bandit_enabled": False},
            "models": {
                "accidental_cloud_gpt": {
                    "provider": "openapi",
                    # NOTE: hosting_tier intentionally omitted!
                    "endpoint": "https://api.openai.com/v1",
                    "model_name": "gpt-4o",
                },
                "local_vllm": {
                    "provider": "openapi",
                    "endpoint": "http://localhost:8000/v1",
                    "model_name": "qwen2.5-coder",
                },
            },
            "roles": {"coding": ["accidental_cloud_gpt", "local_vllm"]},
        }

        store = UsageStore(db_path=tmp_path / "usage.db")
        engine = CostEngine(PricingCatalog(), store)
        bm = BudgetManager(engine, {"enabled": True, "daily_limit": 5.0})
        now = time.monotonic()
        bm._spend_cache["daily"] = (10.0, now)
        bm._spend_cache["monthly"] = (10.0, now)

        fabric = ComputeFabric(cfg, budget_manager=bm)
        # 1. get_local_chain excludes the disguised api.openai.com endpoint!
        assert fabric.get_local_chain("coding") == ["local_vllm"]

        # 2. BudgetCapFilter prunes accidental_cloud_gpt and routes to local_vllm!
        with patch.object(
            OpenAPIProvider,
            "generate_with_usage",
            return_value=GenerateResult(text="vllm ok", input_tokens=10, output_tokens=5),
        ):
            res = fabric.execute("coding", "Write unit test")

        assert res is not None
        assert res.model_id == "local_vllm"
        assert fabric.last_routing_trace is not None
        assert "accidental_cloud_gpt" in fabric.last_routing_trace.pruned


class TestOpenAPIProviderFeatureCompleteness:
    @respx.mock
    def test_extra_headers_extra_body_max_completion_tokens_and_history_schema(self):
        route = respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "qwen-local",
                    "choices": [{"message": {"content": '{"status":"ok"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 25, "completion_tokens": 6},
                },
            )
        )

        prov = OpenAPIProvider({
            "endpoint": "http://localhost:8000/v1",
            "model_name": "qwen-local",
            "api_key": "local-token",
            "extra_headers": {"X-Cluster-Node": "gpu-01"},
            "parameters": {
                "max_completion_tokens": 1024,
                "top_p": 0.9,
                "seed": 42,
                "extra_body": {"repetition_penalty": 1.08},
            },
        })

        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        res = prov.generate_with_history(
            messages=[{"role": "user", "content": "status?"}],
            system_prompt="Return JSON.",
            response_schema=schema,
        )

        assert res.text == '{"status":"ok"}'
        req = route.calls[0].request
        assert req.headers["X-Cluster-Node"] == "gpu-01"
        assert req.headers["Authorization"] == "Bearer local-token"

        body = json.loads(req.content)
        assert body["max_completion_tokens"] == 1024
        assert "max_tokens" not in body
        assert body["top_p"] == 0.9
        assert body["seed"] == 42
        assert body["repetition_penalty"] == 1.08
        assert body["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    @respx.mock
    async def test_stream_handles_empty_choices_usage_chunk_and_stream_options(self):
        sse_lines = "\n".join([
            'data: {"choices":[{"delta":{"content":"Hello "}}]}',
            'data: {"choices":[{"delta":{"content":"vLLM!"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}',
            "data: [DONE]",
            "",
        ])
        route = respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=sse_lines)
        )

        prov = OpenAPIProvider({
            "endpoint": "http://localhost:8000/v1",
            "model_name": "qwen-local",
            "include_stream_usage": True,
        })
        chunks = [tok async for tok in prov.generate_stream("hi")]
        assert chunks == ["Hello ", "vLLM!"]
        sent = json.loads(route.calls[0].request.content)
        assert sent["stream_options"] == {"include_usage": True}

    @respx.mock
    def test_list_models(self):
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "meta-llama/Llama-3.3-70B", "owned_by": "vllm"}]},
            )
        )
        prov = OpenAPIProvider({"endpoint": "http://localhost:8000/v1"})
        models = prov.list_models()
        assert models == [{"id": "meta-llama/Llama-3.3-70B", "model_id": "meta-llama/Llama-3.3-70B", "owned_by": "vllm"}]

