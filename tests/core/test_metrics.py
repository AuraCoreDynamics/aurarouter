import pytest
from aurarouter.config import ConfigLoader
from aurarouter.metrics import DefaultCostOptimizer, OfflineValueEstimator

def test_default_cost_optimizer():
    config = ConfigLoader(allow_missing=True)
    
    # Mocking pricing configuration
    config.set_model("model_expensive", {"provider": "openapi", "cost_per_1m_input": 20.0, "cost_per_1m_output": 40.0})
    config.set_model("model_cheap", {"provider": "openapi", "cost_per_1m_input": 1.0, "cost_per_1m_output": 2.0})
    config.set_model("model_free", {"provider": "ollama"})
    
    metrics = OfflineValueEstimator(config)
    optimizer = DefaultCostOptimizer()
    
    chain = ["model_expensive", "model_free", "model_cheap"]
    ranked = optimizer.rank_models(chain, metrics)
    
    assert ranked == ["model_free", "model_cheap", "model_expensive"]
