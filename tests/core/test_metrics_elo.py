import pytest
from aurarouter.metrics import EloScoringEngine

def test_elo_scoring_engine_initialization():
    engine = EloScoringEngine(initial_rating=1500, k_factor=30)
    assert engine.initial_rating == 1500
    assert engine.k_factor == 30
    assert engine.get_rating("model-a") == 1500

def test_elo_scoring_engine_update_match():
    engine = EloScoringEngine()
    # model-a wins against model-b
    engine.update_match("model-a", "model-b", 1.0)
    
    # model-a should have a higher rating than model-b
    rating_a = engine.get_rating("model-a")
    rating_b = engine.get_rating("model-b")
    
    assert rating_a > 1200.0
    assert rating_b < 1200.0
    
    # rating gain and loss should be symmetric if starting from same rating
    assert rating_a - 1200.0 == 1200.0 - rating_b

def test_elo_scoring_engine_rank_models():
    engine = EloScoringEngine()
    engine.update_match("model-a", "model-c", 1.0) # A beats C
    engine.update_match("model-b", "model-c", 1.0) # B beats C
    engine.update_match("model-a", "model-b", 1.0) # A beats B
    
    ranked = engine.rank_models(["model-a", "model-b", "model-c"])
    assert ranked == ["model-a", "model-b", "model-c"]
