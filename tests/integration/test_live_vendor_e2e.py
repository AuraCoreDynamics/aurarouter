"""Live vendor end-to-end integration test for ComputeFabric.

Automatically discovers available cloud credentials from environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY

If none are present, skips cleanly with zero failures.
When present, routes through ComputeFabric with spend guard max_tokens=16.
"""

from __future__ import annotations

import os
import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric


@pytest.mark.live_vendor
class TestLiveVendorE2E:
    """Canary E2E tests against live vendor models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_keys(self):
        has_any = any(
            os.environ.get(k)
            for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")
        )
        if not has_any:
            pytest.skip("No live cloud vendor API keys set (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)")

    def test_live_cloud_model_routing_smoke(self):
        # Build minimal config containing only configured vendors
        models_cfg = {}
        role_models = []

        if os.environ.get("OPENAI_API_KEY"):
            models_cfg["openai-mini"] = {
                "provider": "openapi",
                "endpoint": "https://api.openai.com/v1",
                "model_name": "gpt-4o-mini",
                "hosting_tier": "cloud",
                "api_key": os.environ["OPENAI_API_KEY"],
            }
            role_models.append("openai-mini")

        if not role_models:
            pytest.skip("No supported direct REST model keys available for live E2E test")

        cfg_data = {
            "version": "2.0",
            "models": models_cfg,
            "roles": {
                "general": role_models,
            },
        }

        cfg_loader = ConfigLoader(allow_missing=True)
        cfg_loader.config = cfg_data
        fabric = ComputeFabric(config=cfg_loader)

        result = fabric.execute(
            role="general",
            prompt="Respond with only the word OK",
            options={"max_tokens": 16},
        )
        assert result is not None
        assert result.text is not None
        assert len(result.text.strip()) > 0

