"""Testing utilities, declarative corpus loader, and simulated providers for AuraRouter."""

from tests.testing.corpus import (
    CORPUS_FIXTURE_PATH,
    execute_scenario_intent_and_routing,
    get_routing_scenario,
    list_routing_scenario_ids,
    load_routing_evaluation_corpus,
    setup_corpus_environment,
)
from tests.testing.simulated_provider import SimulatedProvider

__all__ = [
    "CORPUS_FIXTURE_PATH",
    "SimulatedProvider",
    "execute_scenario_intent_and_routing",
    "get_routing_scenario",
    "list_routing_scenario_ids",
    "load_routing_evaluation_corpus",
    "setup_corpus_environment",
]
