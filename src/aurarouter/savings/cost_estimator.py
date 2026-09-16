"""Dynamic cost estimator infrastructure for AuraRouter."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader
    from aurarouter.auragrid.config_loader import ConfigLoader as GridConfigLoader

logger = logging.getLogger(__name__)


class BaseCostEstimator(ABC):
    """Abstract interface for cost estimation plugins."""
    
    def __init__(self, config: ConfigLoader | GridConfigLoader):
        """Initialize the estimator with access to the global configuration."""
        self.config = config

    @abstractmethod
    def estimate(self, provider_id: str, model_id: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Estimate the cost (in USD) for a generation request.
        
        Args:
            provider_id: The provider ID (e.g. 'google', 'openai', 'ollama').
            model_id: The model ID or name.
            prompt_tokens: The number of tokens in the input prompt.
            completion_tokens: The number of tokens in the generated response.
            
        Returns:
            The estimated cost in USD, or None if the cost cannot be determined.
        """
        ...


class DefaultCostEstimator(BaseCostEstimator):
    """Cost estimator that uses static pricing tables stored in the configuration."""
    
    # Built-in fallback table if none exists in config
    _BUILTIN_TABLE = {
            "google": {
                    "gemini-1.5-pro": {
                            "input_1k": 0.0035,
                            "output_1k": 0.0105
                    },
                    "gemini-1.5-flash": {
                            "input_1k": 0.00035,
                            "output_1k": 0.00105
                    },
                    "gemini-2.0-flash": {
                            "input_1k": 0.0001,
                            "output_1k": 0.0004
                    }
            },
            "openai": {
                    "gpt-4o": {
                            "input_1k": 0.005,
                            "output_1k": 0.015
                    },
                    "gpt-4o-mini": {
                            "input_1k": 0.00015,
                            "output_1k": 0.0006
                    },
                    "o1": {
                            "input_1k": 0.015,
                            "output_1k": 0.06
                    },
                    "o3-mini": {
                            "input_1k": 0.0011,
                            "output_1k": 0.0044
                    }
            },
            "anthropic": {
                    "claude-3-5-sonnet-20241022": {
                            "input_1k": 0.003,
                            "output_1k": 0.015
                    },
                    "claude-3-5-haiku-20241022": {
                            "input_1k": 0.001,
                            "output_1k": 0.005
                    }
            },
            "ollama": {
                    "default": {
                            "input_1k": 0.0,
                            "output_1k": 0.0
                    }
            },
            "bedrock": {
                    "llama-3-8b": {
                            "input_1k": 0.36000000000000004,
                            "output_1k": 0.6
                    },
                    "kimi-k2-5": {
                            "input_1k": 0.465,
                            "output_1k": 5.25
                    },
                    "gpt-oss-20b": {
                            "input_1k": 0.041999999999999996,
                            "output_1k": 0.15
                    },
                    "kimi-k2-thinking": {
                            "input_1k": 0.71,
                            "output_1k": 1.52
                    },
                    "gpt-oss-safeguard-120b": {
                            "input_1k": 0.09000000000000001,
                            "output_1k": 0.3
                    },
                    "minimax-m2-5": {
                            "input_1k": 0.15,
                            "output_1k": 0.7200000000000001
                    },
                    "gemma-3-4b": {
                            "input_1k": 0.030000000000000002,
                            "output_1k": 0.048
                    },
                    "nova-2-0-pro": {
                            "input_1k": 1.33,
                            "output_1k": 5.63
                    },
                    "writer-palmyra-vision-7b": {
                            "input_1k": 0.2625,
                            "output_1k": 0.36000000000000004
                    },
                    "nova-2-0-omni": {
                            "input_1k": 0.2,
                            "output_1k": 2.8
                    },
                    "nova-2-0-lite": {
                            "input_1k": 0.04675,
                            "output_1k": 3.0100000000000002
                    },
                    "nvidia-nemotron-3-super-120b-a12b": {
                            "input_1k": 0.09000000000000001,
                            "output_1k": 0.335
                    },
                    "minimax-m2-1": {
                            "input_1k": 0.18000000000000002,
                            "output_1k": 1.4400000000000002
                    },
                    "glm-4-7": {
                            "input_1k": 0.7200000000000001,
                            "output_1k": 1.7049999999999998
                    },
                    "deepseek-v3-2": {
                            "input_1k": 0.37,
                            "output_1k": 0.925
                    },
                    "gemma-3-12b": {
                            "input_1k": 0.060000000000000005,
                            "output_1k": 0.45
                    },
                    "gpt-oss-safeguard-20b": {
                            "input_1k": 0.0721,
                            "output_1k": 0.35
                    },
                    "qwen3-32b": {
                            "input_1k": 0.18000000000000002,
                            "output_1k": 1.3825
                    },
                    "gemma-3-27b": {
                            "input_1k": 0.23,
                            "output_1k": 1.03
                    },
                    "glm-5": {
                            "input_1k": 2.7125,
                            "output_1k": 1.9200000000000002
                    },
                    "mixtral-8x7b": {
                            "input_1k": 0.5900000000000001,
                            "output_1k": 0.76
                    },
                    "nova-pro": {
                            "input_1k": 0.0,
                            "output_1k": 3.2
                    },
                    "devstral": {
                            "input_1k": 0.7175,
                            "output_1k": 1.2
                    },
                    "llama-3-2-3b": {
                            "input_1k": 0.17,
                            "output_1k": 0.15
                    },
                    "ministral-8b-3-0": {
                            "input_1k": 0.06999999999999999,
                            "output_1k": 0.09000000000000001
                    },
                    "nvidia-nemotron-nano-2": {
                            "input_1k": 0.030000000000000002,
                            "output_1k": 0.18000000000000002
                    },
                    "nova-lite": {
                            "input_1k": 0.015000000000000001,
                            "output_1k": 0.14400000000000002
                    },
                    "gpt-oss-120b": {
                            "input_1k": 0.09000000000000001,
                            "output_1k": 1.085
                    },
                    "nvidia-nemotron-nano-2-vl": {
                            "input_1k": 0.24000000000000002,
                            "output_1k": 0.36000000000000004
                    },
                    "google-gemma-4-26b-a4b": {
                            "input_1k": 0.078,
                            "output_1k": 0.8400000000000001
                    },
                    "voxtral-small-1-0": {
                            "input_1k": 0.21000000000000002,
                            "output_1k": 0.63
                    },
                    "nemotron-nano-3-30b": {
                            "input_1k": 0.05,
                            "output_1k": 0.51
                    },
                    "voxtral-mini-1-0": {
                            "input_1k": 0.030000000000000002,
                            "output_1k": 0.04
                    },
                    "deepseek-v3-1": {
                            "input_1k": 0.696,
                            "output_1k": 1.008
                    },
                    "glm-4-7-flash": {
                            "input_1k": 0.08,
                            "output_1k": 0.24000000000000002
                    },
                    "qwen3-235b-a22b-2507": {
                            "input_1k": 0.38499999999999995,
                            "output_1k": 0.88
                    },
                    "qwen3-vl-235b-a22b": {
                            "input_1k": 1.09,
                            "output_1k": 2.06
                    },
                    "llama-3-1-70b-latency-optimized": {
                            "input_1k": 0.9,
                            "output_1k": 0.9
                    },
                    "google-gemma-4-31b": {
                            "input_1k": 0.16799999999999998,
                            "output_1k": 0.2
                    },
                    "ministral-3b-3-0": {
                            "input_1k": 0.17,
                            "output_1k": 0.12000000000000001
                    },
                    "qwen3-coder-next": {
                            "input_1k": 1.05,
                            "output_1k": 1.24
                    },
                    "qwen3-coder-480b-a35b": {
                            "input_1k": 0.225,
                            "output_1k": 0.9
                    },
                    "qwen3-next-80b-a3b": {
                            "input_1k": 0.16799999999999998,
                            "output_1k": 0.7200000000000001
                    },
                    "nova-micro": {
                            "input_1k": 0.017499999999999998,
                            "output_1k": 0.176
                    },
                    "ministral-14b-3-0": {
                            "input_1k": 0.42000000000000004,
                            "output_1k": 0.3605
                    },
                    "mistral-large-3": {
                            "input_1k": 1.07,
                            "output_1k": 1.82
                    },
                    "qwen3-coder-30b-a3b": {
                            "input_1k": 0.09000000000000001,
                            "output_1k": 0.355
                    },
                    "minimax-m2": {
                            "input_1k": 0.5199999999999999,
                            "output_1k": 2.52
                    },
                    "magistral-small-1-2": {
                            "input_1k": 0.88,
                            "output_1k": 1.82
                    },
                    "google-gemma-4-e2b": {
                            "input_1k": 0.02,
                            "output_1k": 0.16799999999999998
                    },
                    "nova-sonic": {
                            "input_1k": 4.1000000000000005,
                            "output_1k": 14.7
                    },
                    "llama-3-1-405b": {
                            "input_1k": 1.2,
                            "output_1k": 1.2
                    },
                    "mistral-large-2407": {
                            "input_1k": 2.0,
                            "output_1k": 6.0
                    },
                    "llama-4-maverick-17b": {
                            "input_1k": 0.24000000000000002,
                            "output_1k": 0.48500000000000004
                    },
                    "r1": {
                            "input_1k": 1.35,
                            "output_1k": 5.4
                    },
                    "llama-3-3-70b": {
                            "input_1k": 0.36000000000000004,
                            "output_1k": 0.36000000000000004
                    },
                    "mistral-7b": {
                            "input_1k": 0.16,
                            "output_1k": 0.34
                    },
                    "llama-3-70b": {
                            "input_1k": 4.45,
                            "output_1k": 4.2
                    },
                    "llama-3-1-8b": {
                            "input_1k": 0.22,
                            "output_1k": 0.22
                    },
                    "llama-3-1-70b": {
                            "input_1k": 0.36000000000000004,
                            "output_1k": 0.7200000000000001
                    },
                    "llama-4-scout-17b": {
                            "input_1k": 0.17,
                            "output_1k": 0.66
                    },
                    "nova-premier": {
                            "input_1k": 0.625,
                            "output_1k": 6.25
                    },
                    "llama-3-3-70b-custom": {
                            "input_1k": 0.7200000000000001,
                            "output_1k": 0.7200000000000001
                    },
                    "llama-3-2-90b": {
                            "input_1k": 0.36000000000000004,
                            "output_1k": 0.36000000000000004
                    },
                    "llama-3-2-1b": {
                            "input_1k": 0.06499999999999999,
                            "output_1k": 0.12999999999999998
                    },
                    "llama-3-2-11b": {
                            "input_1k": 0.16,
                            "output_1k": 0.08
                    },
                    "nova-pro-latency-optimized": {
                            "input_1k": 1.0,
                            "output_1k": 4.0
                    },
                    "llama-3-1-405b-latency-optimized": {
                            "input_1k": 3.0,
                            "output_1k": 3.0
                    },
                    "mistral-large": {
                            "input_1k": 4.6,
                            "output_1k": 15.6
                    },
                    "nova-sonic-2-0": {
                            "input_1k": 3.0,
                            "output_1k": 3.311
                    },
                    "pixtral-large-25-02": {
                            "input_1k": 2.0,
                            "output_1k": 6.0
                    },
                    "claude-instant": {
                            "input_1k": 0.8,
                            "output_1k": 0.0
                    },
                    "mistral-small": {
                            "input_1k": 1.0,
                            "output_1k": 1.5
                    },
                    "claude-2-0": {
                            "input_1k": 8.0,
                            "output_1k": 0.0
                    },
                    "claude-2-1": {
                            "input_1k": 8.0,
                            "output_1k": 0.0
                    },
                    "claude-3-sonnet": {
                            "input_1k": 3.0,
                            "output_1k": 0.0
                    },
                    "claude-3-haiku": {
                            "input_1k": 0.25,
                            "output_1k": 0.0
                    }
            }
    }
    
    def __init__(self, config: ConfigLoader | GridConfigLoader):
        super().__init__(config)
        self._ensure_table_exists()

    def _get_raw_config(self) -> dict:
        """Helper to get the raw config dict from either loader type."""
        # Standalone ConfigLoader has `.config`
        # GridConfigLoader wraps it and has `._current_loader.config`
        if hasattr(self.config, "config"):
            return self.config.config
        elif hasattr(self.config, "_current_loader") and self.config._current_loader:
            return self.config._current_loader.config
        return {}

    def _ensure_table_exists(self) -> None:
        """Ensure the pricing table is seeded into the configuration if missing."""
        cfg = self._get_raw_config()
        if "pricing_table" not in cfg:
            logger.info("Initializing configuration with default pricing table.")
            # If using GridConfigLoader, push via save_config_change
            if hasattr(self.config, "save_config_change"):
                import asyncio
                try:
                    # Create a task to save it (fire and forget during init)
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.config.save_config_change("pricing_table", self._BUILTIN_TABLE))
                except RuntimeError:
                    # No loop, just mutate local config directly
                    cfg["pricing_table"] = self._BUILTIN_TABLE
                    if self.config._current_loader:
                        self.config._current_loader.save()
            else:
                cfg["pricing_table"] = self._BUILTIN_TABLE
                if hasattr(self.config, "save"):
                    self.config.save()

    def estimate(self, provider_id: str, model_id: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        cfg = self._get_raw_config()
        table = cfg.get("pricing_table", self._BUILTIN_TABLE)
        
        provider_rates = table.get(provider_id)
        if not provider_rates:
            if provider_id == "ollama" or provider_id == "llamacpp":
                return 0.0  # Default local providers to 0
            return None
            
        model_rates = provider_rates.get(model_id, provider_rates.get("default"))
        if not model_rates:
            # Fallback to loose prefix matching (e.g. gpt-4o-2024-05-13 -> gpt-4o)
            for key, rates in provider_rates.items():
                if key != "default" and model_id.startswith(key):
                    model_rates = rates
                    break
                    
        if not model_rates:
            return None
            
        input_cost = (prompt_tokens / 1000.0) * model_rates.get("input_1k", 0.0)
        output_cost = (completion_tokens / 1000.0) * model_rates.get("output_1k", 0.0)
        return input_cost + output_cost


class AzureRetailCostEstimator(BaseCostEstimator):
    """Dynamically fetches Azure OpenAI pricing from the Azure Retail Prices API."""
    
    API_URL = "https://prices.azure.com/api/retail/prices?$filter=serviceFamily eq 'Cognitive Services' and armRegionName eq 'eastus'"
    
    def __init__(self, config: ConfigLoader | GridConfigLoader):
        super().__init__(config)
        self._cache: dict[str, dict[str, float]] = {}

    def update_cache(self) -> dict:
        """Fetch pricing from Azure Retail Prices API."""
        import httpx
        url = self.API_URL
        client = httpx.Client(timeout=30.0)
        pricing = {}
        
        while url:
            try:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"Failed to fetch Azure pricing: {e}")
                break
                
            for item in data.get("Items", []):
                meter = item.get("meterName", "").lower()
                product = item.get("productName", "").lower()
                price = item.get("retailPrice", 0.0)
                
                if "gpt" not in meter and "gpt" not in product:
                    continue
                    
                model_id = "default"
                if "gpt-4o-mini" in meter or "gpt-4o-mini" in product:
                    model_id = "gpt-4o-mini"
                elif "gpt-4o" in meter or "gpt-4o" in product:
                    model_id = "gpt-4o"
                elif "gpt-4" in meter or "gpt-4" in product:
                    model_id = "gpt-4"
                elif "gpt-35" in meter or "gpt-3.5" in product:
                    model_id = "gpt-3.5-turbo"
                    
                if model_id not in pricing:
                    pricing[model_id] = {"input_1k": 0.0, "output_1k": 0.0}
                    
                if "prompt" in meter or "input" in meter:
                    pricing[model_id]["input_1k"] = price * 1000
                elif "completion" in meter or "output" in meter:
                    pricing[model_id]["output_1k"] = price * 1000
                    
            url = data.get("NextPageLink")
            
        self._cache = pricing
        return pricing

    def estimate(self, provider_id: str, model_id: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        if provider_id != "azure_openai":
            return None
            
        if not self._cache:
            # Fallback to default table if cache not initialized
            # To actually run dynamic at runtime without blocking the first request, 
            # this should be triggered via a background task on initialization.
            return DefaultCostEstimator(self.config).estimate(provider_id, model_id, prompt_tokens, completion_tokens)
            
        rates = self._cache.get(model_id, self._cache.get("default"))
        if not rates:
            for key, key_rates in self._cache.items():
                if key != "default" and model_id.startswith(key):
                    rates = key_rates
                    break
        if not rates:
            return None
            
        return (prompt_tokens / 1000.0) * rates.get("input_1k", 0.0) + (completion_tokens / 1000.0) * rates.get("output_1k", 0.0)


class BedrockCostEstimator(BaseCostEstimator):
    """Dynamically fetches AWS Bedrock pricing from the AWS Pricing API."""
    
    API_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/index.json"
    
    def __init__(self, config: ConfigLoader | GridConfigLoader):
        super().__init__(config)
        self._cache: dict[str, dict[str, float]] = {}

    def update_cache(self) -> dict:
        """Fetch pricing from AWS public unauthenticated JSON."""
        import httpx
        try:
            response = httpx.get(self.API_URL, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch AWS Bedrock pricing: {e}")
            return {}
            
        pricing = {}
        products = data.get("products", {})
        terms = data.get("terms", {}).get("OnDemand", {})
        
        for sku, product in products.items():
            attributes = product.get("attributes", {})
            model_name = attributes.get("model")
            if not model_name:
                continue
                
            inference_type = attributes.get("inferenceType")
            if not inference_type:
                continue
                
            term = terms.get(sku)
            if not term:
                continue
                
            price_dimensions = list(term.values())[0].get("priceDimensions", {})
            price_per_unit = list(price_dimensions.values())[0].get("pricePerUnit", {}).get("USD", "0")
            price = float(price_per_unit)
            
            model_id = model_name.lower().replace(" ", "-").replace(".", "-")
            
            if model_id not in pricing:
                pricing[model_id] = {"input_1k": 0.0, "output_1k": 0.0}
                
            if "input" in inference_type.lower():
                pricing[model_id]["input_1k"] = price * 1000
            elif "output" in inference_type.lower():
                pricing[model_id]["output_1k"] = price * 1000
                
        pricing = {k: v for k, v in pricing.items() if v["input_1k"] > 0 or v["output_1k"] > 0}
        self._cache = pricing
        return pricing

    def estimate(self, provider_id: str, model_id: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        if provider_id != "bedrock":
            return None
            
        if not self._cache:
            return DefaultCostEstimator(self.config).estimate(provider_id, model_id, prompt_tokens, completion_tokens)
            
        rates = self._cache.get(model_id)
        if not rates:
            for key, key_rates in self._cache.items():
                if model_id.startswith(key):
                    rates = key_rates
                    break
        if not rates:
            return None
            
        return (prompt_tokens / 1000.0) * rates.get("input_1k", 0.0) + (completion_tokens / 1000.0) * rates.get("output_1k", 0.0)

