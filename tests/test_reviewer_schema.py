"""Tests for reviewer role schema and config (TG-B1)."""

from aurarouter.routing import ReviewResult
from aurarouter.semantic_verbs import BUILTIN_VERBS, resolve_synonym
from aurarouter.config import ConfigLoader


class TestReviewResult:
    def test_round_trip(self):
        """ReviewResult serializes and deserializes correctly."""
        result = ReviewResult(
            verdict="FAIL",
            feedback="Missing error handling",
            correction_hints=["Add try/except around IO call"],
        )
        d = result.to_dict()
        restored = ReviewResult.from_dict(d)
        assert restored.verdict == "FAIL"
        assert restored.feedback == "Missing error handling"
        assert restored.correction_hints == ["Add try/except around IO call"]

    def test_defaults(self):
        """from_dict handles missing fields gracefully."""
        result = ReviewResult.from_dict({})
        assert result.verdict == "PASS"
        assert result.feedback == ""
        assert result.correction_hints == []


class TestReviewerSemanticVerb:
    def test_reviewer_verb_exists(self):
        """The reviewer verb is registered but not required."""
        roles = {v.role for v in BUILTIN_VERBS}
        assert "reviewer" in roles
        reviewer_verb = next(v for v in BUILTIN_VERBS if v.role == "reviewer")
        assert reviewer_verb.required is False

    def test_resolve_synonym_review(self):
        """'review' resolves to 'reviewer' role."""
        assert resolve_synonym("review") == "reviewer"

    def test_resolve_synonym_validate(self):
        """'validate' resolves to 'reviewer' role."""
        assert resolve_synonym("validate") == "reviewer"


class TestMaxReviewIterationsConfig:
    def test_default_is_3(self):
        """Default max_review_iterations is 3 when not configured."""
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}}
        assert config.get_max_review_iterations() == 3

    def test_configured_value(self):
        """Configured max_review_iterations is read correctly."""
        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {},
            "execution": {"max_review_iterations": 5},
        }
        assert config.get_max_review_iterations() == 5

    def test_zero_disables(self):
        """Setting max_review_iterations to 0 disables the review loop."""
        config = ConfigLoader(allow_missing=True)
        config.config = {
            "models": {},
            "execution": {"max_review_iterations": 0},
        }
        assert config.get_max_review_iterations() == 0
