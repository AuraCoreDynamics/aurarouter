"""Tests for the semantic verbs module."""

from aurarouter.semantic_verbs import (
    BUILTIN_VERBS,
    get_known_roles,
    get_required_roles,
    resolve_synonym,
)


class TestResolveSynonym:
    def test_canonical_name_passthrough(self):
        assert resolve_synonym("coding") == "coding"
        assert resolve_synonym("router") == "router"
        assert resolve_synonym("reasoning") == "reasoning"

    def test_builtin_synonym(self):
        assert resolve_synonym("programming") == "coding"
        assert resolve_synonym("classifier") == "router"
        assert resolve_synonym("planner") == "reasoning"
        assert resolve_synonym("architect") == "reasoning"
        assert resolve_synonym("developer") == "coding"

    def test_case_insensitive(self):
        assert resolve_synonym("CODING") == "coding"
        assert resolve_synonym("Programming") == "coding"
        assert resolve_synonym("ROUTER") == "router"

    def test_unknown_passthrough(self):
        assert resolve_synonym("unknown_role") == "unknown_role"
        assert resolve_synonym("foobar") == "foobar"

    def test_custom_verbs_override(self):
        custom = {"my_role": ["alias1", "alias2"]}
        assert resolve_synonym("alias1", custom) == "my_role"
        assert resolve_synonym("alias2", custom) == "my_role"
        assert resolve_synonym("my_role", custom) == "my_role"

    def test_custom_verbs_dont_break_builtins(self):
        custom = {"custom_coding": ["x"]}
        # Built-in synonyms still work when custom verbs are present
        assert resolve_synonym("programming", custom) == "coding"

    def test_custom_takes_priority(self):
        # Custom mapping can override a built-in synonym
        custom = {"special": ["programming"]}
        assert resolve_synonym("programming", custom) == "special"

    def test_whitespace_stripped(self):
        assert resolve_synonym("  coding  ") == "coding"
        assert resolve_synonym(" programming ") == "coding"


class TestKnownRoles:
    def test_returns_list(self):
        roles = get_known_roles()
        assert isinstance(roles, list)
        assert "router" in roles
        assert "reasoning" in roles
        assert "coding" in roles

    def test_order_preserved(self):
        roles = get_known_roles()
        assert roles == [v.role for v in BUILTIN_VERBS]


class TestRequiredRoles:
    def test_required_subset(self):
        required = get_required_roles()
        assert set(required) == {"router", "reasoning", "coding"}

    def test_all_required_are_known(self):
        known = set(get_known_roles())
        for r in get_required_roles():
            assert r in known
