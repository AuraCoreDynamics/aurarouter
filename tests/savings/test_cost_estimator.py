import pytest
from aurarouter.savings.cost_estimator import (
    DefaultCostEstimator,
    AzureRetailCostEstimator,
    BedrockCostEstimator
)

class MockConfig:
    def __init__(self, data):
        self.config = data

def test_default_cost_estimator():
    config = MockConfig({})
    estimator = DefaultCostEstimator(config)
    
    # Test initialization sets pricing table
    assert "pricing_table" in config.config
    
    # Test built-in price
    cost = estimator.estimate("google", "gemini-1.5-flash", 1000, 1000)
    # input_1k: 0.00035, output_1k: 0.00105 => 0.0014
    assert cost == pytest.approx(0.0014)

def test_default_cost_estimator_custom_pricing():
    config = MockConfig({
        "pricing_table": {
            "custom": {
                "model-x": {"input_1k": 1.0, "output_1k": 2.0}
            }
        }
    })
    estimator = DefaultCostEstimator(config)
    
    cost = estimator.estimate("custom", "model-x", 500, 500)
    assert cost == pytest.approx(1.5)

def test_azure_cost_estimator():
    config = MockConfig({})
    estimator = AzureRetailCostEstimator(config)
    
    # Currently a stub
    assert estimator.estimate("azure_openai", "gpt-4o", 1000, 1000) is None
    assert estimator.estimate("google", "gpt-4o", 1000, 1000) is None

def test_bedrock_cost_estimator():
    config = MockConfig({})
    estimator = BedrockCostEstimator(config)
    
    # Currently a stub
    assert estimator.estimate("bedrock", "claude", 1000, 1000) is None
