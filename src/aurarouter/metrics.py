import logging
from typing import Protocol, Optional

from aurarouter.config import ConfigLoader

logger = logging.getLogger("AuraRouter.Metrics")

class MetricProvider(Protocol):
    """Protocol for fetching model metrics."""
    def get_cost_per_1k_tokens(self, model_id: str) -> Optional[float]: ...
    def get_latency_estimate_ms(self, model_id: str) -> Optional[float]: ...

class RoutingOptimizer(Protocol):
    """Protocol for re-ranking a chain of models based on metrics."""
    def rank_models(self, candidate_model_ids: list[str], metrics: MetricProvider) -> list[str]: ...


class OfflineValueEstimator:
    """Estimates metrics offline using configured standard approximations."""
    def __init__(self, config: ConfigLoader):
        self.config = config

    def get_cost_per_1k_tokens(self, model_id: str) -> Optional[float]:
        model_cfg = self.config.get_model_config(model_id)
        inp = model_cfg.get("cost_per_1m_input")
        out = model_cfg.get("cost_per_1m_output")
        if inp is not None and out is not None:
            # Assuming equal input/output token mix for estimation
            return ((inp + out) / 2.0) / 1000.0
        
        # If no pricing is configured, assume local models are free (0.0 cost).
        # Cloud models without pricing would ideally not be routed to, but we return a high default
        # to ensure local models are preferred if dynamic routing is on.
        if model_cfg.get("provider") in ("ollama", "llamacpp", "llamacpp-server"):
            return 0.0
        return 9999.0  # High default cost for unknown cloud models

    def get_latency_estimate_ms(self, model_id: str) -> Optional[float]:
        # Hardcoded approximations
        model_cfg = self.config.get_model_config(model_id)
        if model_cfg.get("provider") in ("ollama", "llamacpp", "llamacpp-server"):
            return 50.0
        return 200.0


class DynamicPricingProvider:
    """Optional plugin that fetches real-time token pricing from external APIs."""
    def __init__(self, config: ConfigLoader):
        self.config = config
        self._offline_fallback = OfflineValueEstimator(config)

    def get_cost_per_1k_tokens(self, model_id: str) -> Optional[float]:
        # Fetching from an external API requires credentials/configuration.
        # For this sprint, we mock the API call failure and fallback gracefully.
        try:
            # TODO: Implement 3rd party OAuth client / pricing feed here
            raise ConnectionError("Pricing API not configured or offline")
        except Exception as e:
            logger.debug(f"Dynamic pricing feed failed, falling back: {e}")
            return self._offline_fallback.get_cost_per_1k_tokens(model_id)

    def get_latency_estimate_ms(self, model_id: str) -> Optional[float]:
        return self._offline_fallback.get_latency_estimate_ms(model_id)


class DefaultCostOptimizer:
    """Ranks candidate models by lowest cost."""
    def rank_models(self, candidate_model_ids: list[str], metrics: MetricProvider) -> list[str]:
        def cost_key(m_id: str) -> float:
            cost = metrics.get_cost_per_1k_tokens(m_id)
            return cost if cost is not None else 999999.0
        
        # stable sort based on cost, keeping original priority if costs are equal
        return sorted(candidate_model_ids, key=cost_key)

class LowestLatencyOptimizer:
    """Ranks candidate models by lowest latency."""
    def rank_models(self, candidate_model_ids: list[str], metrics: MetricProvider) -> list[str]:
        def latency_key(m_id: str) -> float:
            latency = metrics.get_latency_estimate_ms(m_id)
            return latency if latency is not None else 999999.0
        
        return sorted(candidate_model_ids, key=latency_key)


class EloScoringEngine:
    """Tracks and updates ELO ratings for models based on A/B tests or fallback success."""
    
    def __init__(self, initial_rating: float = 1200.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: dict[str, float] = {}

    def get_rating(self, model_id: str) -> float:
        """Get the current rating for a model, defaulting to initial_rating."""
        return self.ratings.get(model_id, self.initial_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for model A against model B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_match(self, model_a: str, model_b: str, score_a: float) -> None:
        """
        Update ratings after a match between model A and model B.
        score_a is 1.0 if A wins, 0.5 for draw, 0.0 if B wins.
        """
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)

        expected_a = self._expected_score(rating_a, rating_b)
        expected_b = self._expected_score(rating_b, rating_a)

        score_b = 1.0 - score_a

        self.ratings[model_a] = rating_a + self.k_factor * (score_a - expected_a)
        self.ratings[model_b] = rating_b + self.k_factor * (score_b - expected_b)

    def rank_models(self, candidate_model_ids: list[str]) -> list[str]:
        """Rank models based on their current ELO rating (highest first)."""
        return sorted(candidate_model_ids, key=self.get_rating, reverse=True)
