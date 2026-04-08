"""AuraXLM MoE advisor contract.

Defines the MCP tool interface that AuraXLM's analyze_route provides,
and the registration expectations for the auraxlm-moe analyzer.
"""

AURAXLM_ANALYZER_SPEC: dict = {
    "analyzer_kind": "moe_ranking",
    "capabilities": ["code", "reasoning", "review", "planning", "domain-expert"],
    "mcp_tool_name": "auraxlm.analyze_route",
}

AURAXLM_ADVISOR_CAPABILITIES = ["routing_advisor"]

ANALYZE_ROUTE_PARAMS = {
    "prompt": {"type": "string", "required": True},
    "intent": {"type": "string", "required": False},
    "candidates": {"type": "array", "items": "ModelMetadata", "required": False},
    "cost_ceiling": {"type": "number", "required": False},
    "latency_ceiling_ms": {"type": "number", "required": False},
    "top_n": {"type": "integer", "required": False, "default": 3},
    # TG4: Optional routing context from AuraRouter's pipeline
    "_aura_routing_context": {
        "type": "object",
        "required": False,
        "description": "AuraRouter pipeline metadata (strategy, confidence_score, "
                       "complexity_score, selected_route, analyzer_chain, intent, "
                       "hard_routed, simulated_cost_avoided). AuraXLM uses this to "
                       "bias expert selection without re-classifying the prompt.",
    },
}

ANALYZE_ROUTE_RESPONSE = {
    "ranked_models": ["model-id-1", "model-id-2"],
    "role": "coding",
    "reasoning": "...",
    "details": [{"model_id": "...", "provider": "...", "capabilities": [], "context_window": 0,
                 "is_local": True, "cost_per_1K_tokens": 0.0, "avg_latency_ms": 0.0}],
    # TG4: Routing context is propagated through and enriched by AuraXLM
    "_aura_routing_context": {
        "strategy": "vector",
        "confidence_score": 0.92,
        "complexity_score": 2,
        "selected_route": "coding",
        "analyzer_chain": ["edge-complexity", "onnx-vector"],
        "intent": "SIMPLE_CODE",
        "hard_routed": False,
        "simulated_cost_avoided": 0.0,
        "metadata": {},
    },
}
